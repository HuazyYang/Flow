#include "RadixSort.h"
#include "Object.h"
#include "ClientHelper.h"

namespace NvFlow {

struct RadixSortImpl : Object, RadixSort {
    NVFLOW_IMPLEMENT_OBJECT_REFERENCE()

    uint64_t getGPUBytesUsed() override;

    RadixSortBuffer getBuffer() override;
    void sort(NvFlowContext *context, const RadixSortParams *params) override;

    // Details
    struct Side {
        NvFlowBuffer *key;
        NvFlowBuffer *val;
        NvFlowBuffer *key4;
        NvFlowBuffer *val4;

        void release() {
            SafeRelease(key);
            SafeRelease(val);
            SafeRelease(key4);
            SafeRelease(val4);
        }
    };

    struct Counter {
        NvFlowBuffer *counter;
        NvFlowBuffer *counter4;

        void release() {
            SafeRelease(counter);
            SafeRelease(counter4);
        }
    };

    RadixSortImpl(NvFlowContext *context, const RadixSortDesc *desc);
    ~RadixSortImpl();

    RadixSortDesc m_desc;
    Side m_front;
    Side m_back;
    Counter m_front_counters;
    Counter m_back_counters;
    NvFlowConstantBuffer *m_constantBuffer;
    NvFlowComputeShader *m_radixSortBlockCS;
    NvFlowComputeShader *m_radixSort1CS;
    NvFlowComputeShader *m_radixSort2CS;
    NvFlowComputeShader *m_radixSort3CS;
};

struct RadixSortCPUImpl : Object, RadixSortCPU {
    NVFLOW_IMPLEMENT_OBJECT_REFERENCE()

    uint64_t getGPUBytesUsed() override;

    RadixSortCPUBuffer getBuffer() override;
    void sort(NvFlowContext *context, const RadixSortCPUParams *params) override;

    // Details
    struct Buffer {
        VectorCached<NvFlowUint2, 1> keyVal;
    };

    RadixSortCPUImpl(const RadixSortCPUDesc *desc);
    ~RadixSortCPUImpl();

    RadixSortCPUDesc m_desc;
    Buffer m_buffer[2];
    unsigned int m_bufferFrontIdx;
    unsigned int m_bufferBackIdx;
    unsigned int m_buckets[256];
    unsigned int m_prefixSum[256];
};

struct RadixSortShaderParams {
    NvFlowUint3 gridDim;
    unsigned int passStart;
    unsigned int numCounters;
};

uint64_t RadixSortCPUImpl::getGPUBytesUsed() {
    return 0;
}

RadixSortCPUBuffer RadixSortCPUImpl::getBuffer() {
    RadixSortCPUBuffer result;
    result.keyVal = m_buffer[m_bufferFrontIdx].keyVal.data();
    return result;
}

void RadixSortCPUImpl::sort(NvFlowContext *context, const RadixSortCPUParams *params) {
    uint32_t numElements = params->numElements;

    for (uint32_t passID = 0; passID < 32; passID += 8) {
        uint32_t shift = passID;
        auto bufFront = m_buffer[m_bufferFrontIdx].keyVal.data();
        auto bufBack = m_buffer[m_bufferBackIdx].keyVal.data();

        ZeroMemory(&m_buckets, sizeof(m_buckets));

        for (uint32_t srcIdx = 0; srcIdx < numElements; ++srcIdx) {
            auto val = bufFront[srcIdx];
            uint32_t bucketIdx = uint8_t(val.x >> shift);
            ++m_buckets[bucketIdx];
        }

        uint32_t totalCount = 0;
        for (uint32_t j = 0; j < countof(m_buckets); ++j) {
            m_prefixSum[j] = totalCount;
            totalCount += m_buckets[j];
        }

        for (uint32_t idx = 0; idx < numElements; ++idx) {
            auto val = bufFront[idx];
            uint32_t bucketIdx = uint8_t(val.x >> shift);
            uint32_t dstIdx = m_prefixSum[bucketIdx]++;
            bufBack[dstIdx] = val;
        }

        swap(m_bufferFrontIdx, m_bufferBackIdx);
    }
}

RadixSortCPUImpl::RadixSortCPUImpl(const RadixSortCPUDesc *desc)
    : m_desc{},
      m_buffer{},
      m_bufferFrontIdx{0},
      m_bufferBackIdx{1},
      m_buckets{},
      m_prefixSum{} {
    m_desc = *desc;
    m_buffer[0].keyVal.resize(m_desc.maxElements);
    m_buffer[1].keyVal.resize(m_desc.maxElements);
}

RadixSortCPUImpl::~RadixSortCPUImpl() {}

#include "radixSortBlockCS.hlsl.h"
#include "radixSort1CS.hlsl.h"
#include "radixSort2CS.hlsl.h"
#include "radixSort3CS.hlsl.h"

RadixSortImpl::RadixSortImpl(NvFlowContext *context, const RadixSortDesc *desc) {
    m_desc = *desc;

    NvFlowBufferDesc bufDesc = {};
    bufDesc.format = eNvFlowFormat_r32_uint;
    bufDesc.dim = m_desc.maxSortBlocks << 10;
    bufDesc.uploadAccess = 0;
    bufDesc.downloadAccess = 0;

    NvFlowBufferViewDesc bufViewDesc = {};
    bufViewDesc.format = eNvFlowFormat_r32g32b32a32_uint;

    m_front.key = NvFlowCreateBuffer(context, &bufDesc);
    m_front.val = NvFlowCreateBuffer(context, &bufDesc);
    m_back.key = NvFlowCreateBuffer(context, &bufDesc);
    m_back.val = NvFlowCreateBuffer(context, &bufDesc);
    m_front.key4 = NvFlowCreateBufferView(context, m_front.key, &bufViewDesc);
    m_front.val4 = NvFlowCreateBufferView(context, m_front.val, &bufViewDesc);
    m_back.key4 = NvFlowCreateBufferView(context, m_back.key, &bufViewDesc);
    m_back.val4 = NvFlowCreateBufferView(context, m_back.val, &bufViewDesc);

    bufDesc.dim = 32 * m_desc.maxSortBlocks;
    m_front_counters.counter = NvFlowCreateBuffer(context, &bufDesc);
    m_back_counters.counter = NvFlowCreateBuffer(context, &bufDesc);
    m_front_counters.counter4 =
        NvFlowCreateBufferView(context, m_front_counters.counter, &bufViewDesc);
    m_back_counters.counter4 =
        NvFlowCreateBufferView(context, m_back_counters.counter, &bufViewDesc);

    NvFlowConstantBufferDesc cbDesc = {};
    cbDesc.sizeInBytes = sizeof(RadixSortShaderParams);
    cbDesc.uploadAccess = 1;
    m_constantBuffer = NvFlowCreateConstantBuffer(context, &cbDesc);

    auto createShader = [context](const BYTE *cs, uint64_t cs_length,
                                  const wchar_t *label) {
        NvFlowComputeShaderDesc desc = {};
        desc.cs = cs;
        desc.cs_length = cs_length;
        desc.label = label;
        return NvFlowCreateComputeShader(context, &desc);
    };

    m_radixSortBlockCS = createShader(NVFLOW_CREATE_SHADER_ARGS(radixSortBlockCS));
    m_radixSort1CS = createShader(NVFLOW_CREATE_SHADER_ARGS(radixSort1CS));
    m_radixSort2CS = createShader(NVFLOW_CREATE_SHADER_ARGS(radixSort2CS));
    m_radixSort3CS = createShader(NVFLOW_CREATE_SHADER_ARGS(radixSort3CS));
}

uint64_t RadixSortImpl::getGPUBytesUsed() {
    return 0;
}

RadixSortBuffer RadixSortImpl::getBuffer() {
    RadixSortBuffer result;
    result.key = m_front.key;
    result.val = m_front.val;
    return result;
}

void RadixSortImpl::sort(NvFlowContext *context, const RadixSortParams *params) {
    if (params->numSortBlocks == 1) {
        NvFlowDispatchParams dparams = {};
        dparams.shader = m_radixSortBlockCS;
        dparams.gridDim = make_dim(1);
        dparams.rootConstantBuffer = m_constantBuffer;
        dparams.readOnly[0] = NvFlowBufferGetResource(m_front.key4);
        dparams.readOnly[1] = NvFlowBufferGetResource(m_front.val4);
        dparams.readWrite[0] = NvFlowBufferGetResourceRW(m_back.key4);
        dparams.readWrite[1] = NvFlowBufferGetResourceRW(m_back.val4);
        NvFlowContextDispatch(context, &dparams);
        swap(m_front, m_back);
    } else if (params->numSortBlocks >= 1) {
        uint32_t numBlocks = params->numSortBlocks;
        for (uint32_t passID = 0; passID < 32; passID += 4) {
            auto mapped =
                (RadixSortShaderParams *)NvFlowConstantBufferMap(context, m_constantBuffer);
            if (mapped) {
                mapped->gridDim = make_uint3(numBlocks, 1, 1);
                mapped->numCounters = 16 * numBlocks;
                mapped->passStart = passID;
                NvFlowConstantBufferUnmap(context, m_constantBuffer);
            }

            NvFlowDispatchParams dparams = {};
            dparams.shader = m_radixSort1CS;
            dparams.gridDim = make_dim(numBlocks, 1, 1);
            dparams.rootConstantBuffer = m_constantBuffer;
            dparams.readOnly[0] = NvFlowBufferGetResource(m_front.key4);
            dparams.readWrite[0] = NvFlowBufferGetResourceRW(m_front_counters.counter);
            NvFlowContextDispatch(context, &dparams);

            dparams.shader = m_radixSort2CS;
            dparams.gridDim = make_dim(2, 1, 1);
            dparams.readOnly[0] = NvFlowBufferGetResource(m_front_counters.counter4);
            dparams.readWrite[0] = NvFlowBufferGetResourceRW(m_back_counters.counter4);
            NvFlowContextDispatch(context, &dparams);

            swap(m_front_counters, m_back_counters);

            dparams.shader = m_radixSort3CS;
            dparams.gridDim = make_dim(numBlocks, 1, 1);
            dparams.readOnly[0] = NvFlowBufferGetResource(m_front.key4);
            dparams.readOnly[1] = NvFlowBufferGetResource(m_front.val4);
            dparams.readOnly[2] = NvFlowBufferGetResource(m_front_counters.counter);
            dparams.readWrite[0] = NvFlowBufferGetResourceRW(m_back.key);
            dparams.readWrite[1] = NvFlowBufferGetResourceRW(m_back.val);
            NvFlowContextDispatch(context, &dparams);

            swap(m_front, m_back);
        }
    }
}

RadixSortImpl::~RadixSortImpl() {
    m_front.release();
    m_back.release();
    m_front_counters.release();
    m_back_counters.release();
    SafeRelease(m_constantBuffer);
    SafeRelease(m_radixSortBlockCS);
    SafeRelease(m_radixSort1CS);
    SafeRelease(m_radixSort2CS);
    SafeRelease(m_radixSort3CS);
}

RadixSort *createRadixSort(NvFlowContext *context, const RadixSortDesc *desc) {
    return new RadixSortImpl(context, desc);
}

RadixSortCPU *createRadixSortCPU(const RadixSortCPUDesc *desc) {
    return new RadixSortCPUImpl(desc);
}

}  // namespace NvFlow