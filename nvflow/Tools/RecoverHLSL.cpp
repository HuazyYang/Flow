#include <filesystem>
#include <cstdarg>
#include <cassert>
#include <fstream>
#include <string>
#include <vector>

#include "dxbc_spdb.h"

namespace DXBC {
struct Hash {
    uint8_t Digest[16];
};

enum class HashFlags : uint32_t {
    None = 0,            // No flags defined.
    IncludesSource = 1,  // This flag indicates that the shader hash was computed
                         // taking into account source information (-Zss)
};

struct ShaderHash {
    uint32_t Flags;  // dxbc::HashFlags
    uint8_t Digest[16];

    bool isPopulated();
};

struct ContainerVersion {
    uint16_t Major;
    uint16_t Minor;
};

struct Header {
    uint8_t Magic[4];  // "DXBC"
    Hash FileHash;
    ContainerVersion Version;
    uint32_t FileSize;
    uint32_t PartCount;
    // Structure is followed by part offsets: uint32_t PartOffset[PartCount];
    // The offset is to a PartHeader, which is followed by the Part Data.
};

/// Use this type to describe the size and type of a DXIL container part.
struct PartHeader {
    uint8_t Name[4];
    uint32_t Size;
    // Structure is followed directly by part data: uint8_t PartData[PartSize].
};

const byte *GetSPDBHeaderAddress(const byte *dxbc, uint32_t *spdblength) {
    const Header *header = (const Header *)dxbc;
    const uint32_t *partOffsets = (const uint32_t *)(header + 1);

    const PartHeader *part;

    for (uint32_t i = 0; i < header->PartCount; ++i) {
        part = (const PartHeader *)(dxbc + partOffsets[i]);

        if (memcmp(part->Name, "SPDB", 4) == 0) {
            *spdblength = part->Size;
            return (const byte *)(part + 1);
        }
    }

    return nullptr;
}

rdcarray<ShaderSourceFile> RecoverHLSLSourceFormDXBC(const byte *dxbc, size_t srcLen) {
    uint32_t dataLength;
    auto data = GetSPDBHeaderAddress(dxbc, &dataLength);

    SPDBChunk pdbChunk((byte *)data, dataLength);
    return std::move(pdbChunk.Files);
}

}  // namespace DXBC

void usage(const char *prog) {
    fprintf(stdout,
            "usage: %s [-d output directory] [-Fs exported filenames...] [-h] <DXBC binary "
            "file>\n",
            prog);
    fprintf(stdout,
            "  -d           Output directory component of exported file(s)\n"
            "  -Fs          Filename component of exported file(s)\n"
            "  -h           Show this usage message\n");
    return;
}

void dump_invalid_arg(const char *arg, const char *help_info, ...) {
    fprintf(stderr, "unknown argument: %s\n", arg);
    if (help_info && help_info[0]) {
        va_list ap;
        va_start(ap, help_info);
        vfprintf(stderr, help_info, ap);
        va_end(ap);
    }
}

struct InputArgs {
    const char *dxbc_filepath;
    const char *output_dir;
    std::vector<const char *> output_filenames;
};

int parse_command_line(int argc, char *argv[], InputArgs &args) {
    args.dxbc_filepath = nullptr;
    args.output_dir = nullptr;
    args.output_filenames.clear();

    int i;
    for (i = 1; i < argc; ++i) {
        if (argv[i][0] == '-') {
            if (argv[i][1] == 'd') {
                if (argv[i][2] == 0) {
                    if (i + 1 < argc) {
                        args.output_dir = argv[++i];
                    } else {
                        fprintf(stderr, "-d option require a argument\n");
                        return -1;
                    }
                } else
                    args.output_dir = argv[i] + 2;
            } else if (strlen(argv[i] + 1) >= 2 && strncmp(argv[i] + 1, "Fs", 2) == 0) {
                if (argv[i][3] == 0) {
                    if (i + 1 < argc) {
                        args.output_filenames.push_back(argv[++i]);
                    } else {
                        fprintf(stderr, "-Fs option require a argument\n");
                        return -1;
                    }
                } else
                    args.output_filenames.push_back(argv[i] + 3);
            } else if (strcmp(argv[i] + 1, "h") == 0) {
                usage(argv[0]);
                return -1;
            } else {
                dump_invalid_arg(argv[i], nullptr);
                return -1;
            }
        } else {
            if (args.dxbc_filepath) {
                dump_invalid_arg(
                    argv[i],
                    "DXBC binary file specfied more than one time, the lastest file is: %s",
                    args.dxbc_filepath);
                return -1;
            } else
                args.dxbc_filepath = argv[i];
        }
    }

    if (i != argc) {
        for (; i < argc; ++i)
            dump_invalid_arg(argv[i], nullptr);
        usage(argv[0]);
        return -1;
    }

    if (!args.dxbc_filepath) {
        fprintf(stderr, "DXBC file must be specified!\n");
        return -1;
    }

    return 0;
}

static std::filesystem::path resolve_output_path(
    const char *filename, const std::filesystem::path &preset_output_dir) {
    std::filesystem::path filepath{filename};
    if (filepath.is_relative()) filepath = preset_output_dir / filepath;
    filepath = std::filesystem::weakly_canonical(filepath);
    return filepath;
}

int main(int argc, char *argv[]) {
    InputArgs args;
    if (parse_command_line(argc, argv, args)) return -1;

    std::vector<uint8_t> dxbc_buffer;
    {
        std::ifstream fin(args.dxbc_filepath, std::ios::binary);
        if (!fin) {
            fprintf(stderr, "Failed to open file %s\n", args.dxbc_filepath);
            return -1;
        }
        fin.seekg(0, std::ios::end);
        dxbc_buffer.resize(fin.tellg());
        fin.seekg(0, std::ios::beg);
        fin.read((char *)dxbc_buffer.data(), dxbc_buffer.size());
        fin.close();
    }

    auto shader_srcs =
        DXBC::RecoverHLSLSourceFormDXBC(dxbc_buffer.data(), dxbc_buffer.size());

    std::filesystem::path output_dir{args.output_dir
                                         ? std::filesystem::canonical(args.output_dir)
                                         : std::filesystem::current_path()};

    struct OutputFileInfo {
        std::filesystem::path filepath;
        std::filesystem::path mapped_filepath;
        const DXBC::ShaderSourceFile *source;
    };
    std::vector<OutputFileInfo> output_infos(args.output_filenames.size());
    for (int i = 0; i < output_infos.size(); ++i) {
        output_infos[i] = OutputFileInfo{
            resolve_output_path(args.output_filenames[i], output_dir), "", nullptr};
    }

    for (auto &src : shader_srcs) {
        // Dump shader source file path
        fprintf(stdout, "Source: %s\n", src.filename.c_str());

        std::filesystem::path mapped_filepath{src.filename};
        std::string mapped_filename = mapped_filepath.filename().string();
        auto it_output =
            std::find_if(output_infos.begin(), output_infos.end(),
                         [mapped_filename](const OutputFileInfo &out_info) {
                             return _stricmp(out_info.filepath.filename().string().c_str(),
                                             mapped_filename.c_str()) == 0;
                         });
        if (it_output != output_infos.end()) {
            it_output->mapped_filepath = mapped_filepath;
            it_output->source = &src;
        }
    }

    for (auto &output_info : output_infos) {
        if (output_info.source) {
            fprintf(stdout, "Write: %s <- %s\n", output_info.filepath.string().c_str(),
                    output_info.mapped_filepath.string().c_str());

            std::ofstream fout{output_info.filepath, std::ios::binary};

            fout.write((const char *)output_info.source->contents.data(),
                       output_info.source->contents.size());
        } else {
            fprintf(stdout, "Warning: %s is not included in %s SPDB content\n",
                    output_info.filepath.string().c_str(), args.dxbc_filepath);
        }
    }

    return 0;
}