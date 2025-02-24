#include <filesystem>
#include <cassert>
#include <fstream>
#include <string>
#include <iostream>
#include <string_view>
#include <vector>
#include <array>

#include "dxbc_spdb.h"

#include "NvFlow__g_sparseShiftCS.h"

namespace fs = std::filesystem;

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

int RecoverHLSLSourceFormDXBC(const byte *dxbc, size_t srcLen,
                              const std::array<std::string, 2> &fileNames,
                              const fs::path &outputDir) {
    uint32_t dataLength;
    auto data = GetSPDBHeaderAddress(dxbc, &dataLength);

    SPDBChunk pdbChunk((byte *)data, dataLength);

    std::array<int, 2> checked;
    for (uint32_t i = 0; i < fileNames.size(); ++i) {
        checked[i] = fileNames[i].empty() ? 1 : 0;
    }

    for (auto &file : pdbChunk.Files) {
        auto originFileName = fs::path(file.filename).filename();
        auto originFileNameNoExt = originFileName.stem().string();
        auto ext = originFileName.extension();

        for (size_t i = 0; i < fileNames.size(); ++i) {
            auto &reqFileName = fileNames[i];
            if (!reqFileName.empty() &&
                _stricmp(reqFileName.data(), originFileNameNoExt.data()) == 0) {
                auto outputFPath = outputDir / (fs::path(reqFileName) += ext);
                std::ofstream fout(outputFPath);
                fout.write(file.contents.data(), file.contents.size());
                checked[i] = 1;
            }
        }
    }

    for (size_t i = 0; i < checked.size(); ++i)
        if (!checked[i]) return 1;

    return 0;
}

}  // namespace DXBC

int main(int args, char *argv[]) {
    if (args < 2) {
        printf("usage: %s <Shader output directory>\n", argv[0]);
        return -1;
    }

    const char *exportDir = argv[1];

    struct ExportShaderFileInfo {
        const DXBC::byte *dxbc;
        size_t dxbcLength;
        std::array<std::string, 2> fileNames;
    } exportShaderFileInfos[] = {
#define DXBC_INCLUDE_CONTENT(var) var, sizeof(var)
        // clang-format off
        {
            DXBC_INCLUDE_CONTENT(NvFlow__g_sparseShiftCS),
            {
                "sparseShiftCS"
            }
        },
    // clang-format on
#undef DXBC_INCLUDE_CONTENT
    };

    for (auto &exportShaderInfo : exportShaderFileInfos) {
        int rc = DXBC::RecoverHLSLSourceFormDXBC(exportShaderInfo.dxbc,
                                                 exportShaderInfo.dxbcLength,
                                                 exportShaderInfo.fileNames, exportDir);

        assert(rc == 0);
    }
    return 0;
}