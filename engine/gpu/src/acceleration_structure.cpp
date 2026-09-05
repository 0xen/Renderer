#include "rend/gpu/acceleration_structure.h"

#include "rend/core/log.h"
#include "rend/core/profile.h"
#include "rend/gpu/buffer.h"
#include "rend/gpu/device.h"

#include <volk.h>

#include <algorithm>
#include <cstring>
#include <format>
#include <vector>

namespace rend::gpu {

namespace {

VkDeviceAddress bufferAddress(const Device& device, VkBuffer buffer) {
    VkBufferDeviceAddressInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    info.buffer = buffer;
    return vkGetBufferDeviceAddress(device.handle(), &info);
}

// Synchronous one-shot on the graphics queue (load-time builds only).
Result<void> submitOnce(const Device& device, void (*record)(VkCommandBuffer, const void*),
                        const void* userData) {
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    poolInfo.queueFamilyIndex = device.graphicsQueue().familyIndex;
    VkCommandPool pool = VK_NULL_HANDLE;
    if (VkResult r = vkCreateCommandPool(device.handle(), &poolInfo, nullptr, &pool);
        r != VK_SUCCESS) {
        return Error{std::format("vkCreateCommandPool (AS build) failed ({})", static_cast<int>(r))};
    }

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = pool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(device.handle(), &allocInfo, &cmd);

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin);
    record(cmd, userData);
    vkEndCommandBuffer(cmd);

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    vkCreateFence(device.handle(), &fenceInfo, nullptr, &fence);

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    Result<void> out{};
    if (VkResult r = vkQueueSubmit(device.graphicsQueue().queue, 1, &submit, fence);
        r != VK_SUCCESS) {
        out = Error{std::format("vkQueueSubmit (AS build) failed ({})", static_cast<int>(r))};
    } else if (VkResult r2 = vkWaitForFences(device.handle(), 1, &fence, VK_TRUE,
                                             10'000'000'000ull);
               r2 != VK_SUCCESS) {
        out = Error{std::format("AS build fence wait failed ({})", static_cast<int>(r2))};
    }
    vkDestroyFence(device.handle(), fence, nullptr);
    vkDestroyCommandPool(device.handle(), pool, nullptr);
    return out;
}

struct BuildJob {
    VkAccelerationStructureBuildGeometryInfoKHR build{};
    const VkAccelerationStructureBuildRangeInfoKHR* ranges = nullptr;
};

void recordBuild(VkCommandBuffer cmd, const void* userData) {
    const auto* job = static_cast<const BuildJob*>(userData);
    vkCmdBuildAccelerationStructuresKHR(cmd, 1, &job->build, &job->ranges);
}

struct Built {
    VkAccelerationStructureKHR handle = VK_NULL_HANDLE;
    std::unique_ptr<Buffer> storage;
    std::unique_ptr<Buffer> updateScratch;
};

// Converts caller geometry ranges into the Vulkan build structures —
// shared by the load-time build and per-frame refits (which pass the same
// ranges with different vertex addresses).
void fillTriangleGeometries(const Device& device,
                            std::span<const AccelerationStructure::TriangleGeometry> geometries,
                            std::vector<VkAccelerationStructureGeometryKHR>& geos,
                            std::vector<VkAccelerationStructureBuildRangeInfoKHR>& ranges) {
    geos.resize(geometries.size());
    ranges.resize(geometries.size());
    for (std::size_t i = 0; i < geometries.size(); ++i) {
        const AccelerationStructure::TriangleGeometry& g = geometries[i];
        const VkDeviceAddress base = bufferAddress(device, g.buffer->handle());
        auto& geo = geos[i];
        geo = {};
        geo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        geo.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        geo.flags = g.opaque ? VK_GEOMETRY_OPAQUE_BIT_KHR : 0;
        auto& tris = geo.geometry.triangles;
        tris.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        tris.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        tris.vertexData.deviceAddress = base + g.vertexOffset;
        tris.vertexStride = g.vertexStride;
        tris.maxVertex = g.vertexCount > 0 ? g.vertexCount - 1 : 0;
        tris.indexType = VK_INDEX_TYPE_UINT32;
        tris.indexData.deviceAddress = base + g.indexOffset;
        ranges[i] = {};
        ranges[i].primitiveCount = g.indexCount / 3;
    }
}

// Shared tail: size the build, allocate storage + scratch, create the AS
// handle, run the build. allowUpdate also sizes and keeps the UPDATE
// scratch so the structure can be refitted later.
Result<Built> buildCommon(
    const Device& device, VkAccelerationStructureTypeKHR type,
    VkAccelerationStructureBuildGeometryInfoKHR& build,
    const std::vector<VkAccelerationStructureBuildRangeInfoKHR>& ranges,
    const std::vector<std::uint32_t>& primitiveCounts, bool allowUpdate) {
    build.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    build.type = type;
    build.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                  (allowUpdate ? VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR : 0);
    build.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;

    VkAccelerationStructureBuildSizesInfoKHR sizes{};
    sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    vkGetAccelerationStructureBuildSizesKHR(device.handle(),
                                            VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                            &build, primitiveCounts.data(), &sizes);

    auto storageResult = Buffer::create(
        device, {
                    .size = sizes.accelerationStructureSize,
                    .usage = 0x00100000 /*ACCELERATION_STRUCTURE_STORAGE*/ |
                             kUsageShaderDeviceAddress,
                    .location = MemoryLocation::DeviceLocal,
                });
    if (!storageResult) {
        return Error{std::format("AS storage: {}", storageResult.error().message)};
    }
    auto scratchResult = Buffer::create(device, {
                                                    .size = sizes.buildScratchSize,
                                                    .usage = kUsageStorage |
                                                             kUsageShaderDeviceAddress,
                                                    .location = MemoryLocation::DeviceLocal,
                                                });
    if (!scratchResult) {
        return Error{std::format("AS scratch: {}", scratchResult.error().message)};
    }
    std::unique_ptr<Buffer> updateScratch;
    if (allowUpdate) {
        auto updateResult =
            Buffer::create(device, {
                                       .size = std::max<std::uint64_t>(sizes.updateScratchSize, 4),
                                       .usage = kUsageStorage | kUsageShaderDeviceAddress,
                                       .location = MemoryLocation::DeviceLocal,
                                   });
        if (!updateResult) {
            return Error{std::format("AS update scratch: {}", updateResult.error().message)};
        }
        updateScratch = std::move(updateResult).value();
    }

    VkAccelerationStructureCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    createInfo.buffer = storageResult.value()->handle();
    createInfo.size = sizes.accelerationStructureSize;
    createInfo.type = type;
    VkAccelerationStructureKHR handle = VK_NULL_HANDLE;
    if (VkResult r =
            vkCreateAccelerationStructureKHR(device.handle(), &createInfo, nullptr, &handle);
        r != VK_SUCCESS) {
        return Error{std::format("vkCreateAccelerationStructureKHR failed ({})",
                                 static_cast<int>(r))};
    }

    build.dstAccelerationStructure = handle;
    build.scratchData.deviceAddress = bufferAddress(device, scratchResult.value()->handle());

    BuildJob job{build, ranges.data()};
    if (auto r = submitOnce(device, recordBuild, &job); !r) {
        vkDestroyAccelerationStructureKHR(device.handle(), handle, nullptr);
        return r.error();
    }
    return Built{handle, std::move(storageResult).value(), std::move(updateScratch)};
}

} // namespace

Result<std::unique_ptr<AccelerationStructure>> AccelerationStructure::buildBottomLevel(
    const Device& device, std::span<const TriangleGeometry> geometries, bool allowUpdate) {
    REND_PROFILE_ZONE("BuildBLAS");
    if (geometries.empty()) {
        return Error{"BLAS build needs at least one geometry"};
    }
    std::vector<VkAccelerationStructureGeometryKHR> geos;
    std::vector<VkAccelerationStructureBuildRangeInfoKHR> ranges;
    fillTriangleGeometries(device, geometries, geos, ranges);
    std::vector<std::uint32_t> primitiveCounts(geometries.size());
    for (std::size_t i = 0; i < ranges.size(); ++i) {
        primitiveCounts[i] = ranges[i].primitiveCount;
    }

    VkAccelerationStructureBuildGeometryInfoKHR build{};
    build.geometryCount = static_cast<std::uint32_t>(geos.size());
    build.pGeometries = geos.data();
    auto result = buildCommon(device, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, build,
                              ranges, primitiveCounts, allowUpdate);
    if (!result) {
        return result.error();
    }
    auto as = std::unique_ptr<AccelerationStructure>(new AccelerationStructure());
    as->device_ = &device;
    as->as_ = result.value().handle;
    as->storage_ = std::move(result.value().storage);
    as->updateScratch_ = std::move(result.value().updateScratch);
    log::info("BLAS built: {} geometries{}", geos.size(), allowUpdate ? " (updatable)" : "");
    return as;
}

Result<std::unique_ptr<AccelerationStructure>> AccelerationStructure::buildTopLevel(
    const Device& device, std::span<const Instance> instances, bool allowUpdate) {
    REND_PROFILE_ZONE("BuildTLAS");
    if (instances.empty()) {
        return Error{"TLAS build needs at least one instance"};
    }
    // Instance array in a host-visible buffer the build reads directly.
    std::vector<VkAccelerationStructureInstanceKHR> data(instances.size());
    for (std::size_t i = 0; i < instances.size(); ++i) {
        auto& inst = data[i];
        inst = {};
        inst.transform.matrix[0][0] = 1.0f;
        inst.transform.matrix[1][1] = 1.0f;
        inst.transform.matrix[2][2] = 1.0f;
        inst.instanceCustomIndex = instances[i].customIndex;
        inst.mask = 0xFF;
        inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;

        VkAccelerationStructureDeviceAddressInfoKHR addressInfo{};
        addressInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
        addressInfo.accelerationStructure = instances[i].blas->handle();
        inst.accelerationStructureReference =
            vkGetAccelerationStructureDeviceAddressKHR(device.handle(), &addressInfo);
    }
    auto instanceBufferResult = Buffer::create(
        device, {
                    .size = data.size() * sizeof(VkAccelerationStructureInstanceKHR),
                    .usage = kUsageAccelBuildInput | kUsageShaderDeviceAddress,
                    .location = MemoryLocation::HostVisible,
                });
    if (!instanceBufferResult) {
        return Error{std::format("TLAS instances: {}", instanceBufferResult.error().message)};
    }
    auto& instanceBuffer = instanceBufferResult.value();
    std::memcpy(instanceBuffer->mapped(), data.data(),
                data.size() * sizeof(VkAccelerationStructureInstanceKHR));

    VkAccelerationStructureGeometryKHR geo{};
    geo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geo.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geo.geometry.instances.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    geo.geometry.instances.data.deviceAddress =
        bufferAddress(device, instanceBuffer->handle());

    std::vector<VkAccelerationStructureBuildRangeInfoKHR> ranges(1);
    ranges[0].primitiveCount = static_cast<std::uint32_t>(data.size());
    const std::vector<std::uint32_t> primitiveCounts{
        static_cast<std::uint32_t>(data.size())};

    VkAccelerationStructureBuildGeometryInfoKHR build{};
    build.geometryCount = 1;
    build.pGeometries = &geo;
    auto result = buildCommon(device, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, build, ranges,
                              primitiveCounts, allowUpdate);
    if (!result) {
        return result.error();
    }
    auto as = std::unique_ptr<AccelerationStructure>(new AccelerationStructure());
    as->device_ = &device;
    as->as_ = result.value().handle;
    as->storage_ = std::move(result.value().storage);
    as->updateScratch_ = std::move(result.value().updateScratch);
    // Refits reread the instances, so an updatable TLAS keeps the buffer.
    if (allowUpdate) {
        as->instances_ = std::move(instanceBuffer);
        as->instanceCount_ = static_cast<std::uint32_t>(data.size());
    }
    log::info("TLAS built: {} instances{}", data.size(), allowUpdate ? " (updatable)" : "");
    return as;
}

void AccelerationStructure::recordRefit(VkCommandBuffer cmd,
                                        std::span<const TriangleGeometry> geometries) const {
    std::vector<VkAccelerationStructureGeometryKHR> geos;
    std::vector<VkAccelerationStructureBuildRangeInfoKHR> ranges;
    fillTriangleGeometries(*device_, geometries, geos, ranges);

    VkAccelerationStructureBuildGeometryInfoKHR build{};
    build.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    build.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    build.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                  VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    build.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
    build.srcAccelerationStructure = as_;
    build.dstAccelerationStructure = as_;
    build.geometryCount = static_cast<std::uint32_t>(geos.size());
    build.pGeometries = geos.data();
    build.scratchData.deviceAddress = bufferAddress(*device_, updateScratch_->handle());

    const VkAccelerationStructureBuildRangeInfoKHR* rangePtr = ranges.data();
    vkCmdBuildAccelerationStructuresKHR(cmd, 1, &build, &rangePtr);
}

void AccelerationStructure::recordRefit(VkCommandBuffer cmd) const {
    VkAccelerationStructureGeometryKHR geo{};
    geo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geo.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geo.geometry.instances.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    geo.geometry.instances.data.deviceAddress = bufferAddress(*device_, instances_->handle());

    VkAccelerationStructureBuildGeometryInfoKHR build{};
    build.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    build.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    build.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                  VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    build.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
    build.srcAccelerationStructure = as_;
    build.dstAccelerationStructure = as_;
    build.geometryCount = 1;
    build.pGeometries = &geo;
    build.scratchData.deviceAddress = bufferAddress(*device_, updateScratch_->handle());

    VkAccelerationStructureBuildRangeInfoKHR range{};
    range.primitiveCount = instanceCount_;
    const VkAccelerationStructureBuildRangeInfoKHR* rangePtr = &range;
    vkCmdBuildAccelerationStructuresKHR(cmd, 1, &build, &rangePtr);
}

AccelerationStructure::~AccelerationStructure() {
    if (device_ && as_ != VK_NULL_HANDLE) {
        vkDestroyAccelerationStructureKHR(device_->handle(), as_, nullptr);
    }
}

} // namespace rend::gpu
