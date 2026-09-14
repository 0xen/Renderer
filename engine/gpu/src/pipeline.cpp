#include "rend/gpu/pipeline.h"

#include "rend/core/log.h"
#include "rend/gpu/device.h"
#include "rend/gpu/shader.h"

#include <volk.h>

#include <array>
#include <vector>

namespace rend::gpu {

Result<std::unique_ptr<Pipeline>> Pipeline::createCompute(const Device& device,
                                                          const ComputePipelineDesc& desc) {
    if (!desc.shader) {
        return Error{"Compute pipeline needs a shader"};
    }

    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.size = desc.pushConstantBytes;

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    if (desc.pushConstantBytes > 0) {
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &pushRange;
    }
    if (desc.descriptorLayout != nullptr) {
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &desc.descriptorLayout;
    }

    VkPipelineLayout layout = VK_NULL_HANDLE;
    if (VkResult r = vkCreatePipelineLayout(device.handle(), &layoutInfo, nullptr, &layout);
        r != VK_SUCCESS) {
        return Error{std::format("vkCreatePipelineLayout failed ({})", static_cast<int>(r))};
    }

    VkComputePipelineCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    info.stage.module = desc.shader->handle();
    info.stage.pName = desc.entryPoint;
    info.layout = layout;

    VkPipeline handle = VK_NULL_HANDLE;
    if (VkResult r =
            vkCreateComputePipelines(device.handle(), VK_NULL_HANDLE, 1, &info, nullptr, &handle);
        r != VK_SUCCESS) {
        vkDestroyPipelineLayout(device.handle(), layout, nullptr);
        return Error{std::format("vkCreateComputePipelines failed ({})", static_cast<int>(r))};
    }

    auto pipeline = std::unique_ptr<Pipeline>(new Pipeline());
    pipeline->device_ = &device;
    pipeline->layout_ = layout;
    pipeline->pipeline_ = handle;
    pipeline->compute_ = true;
    log::info("Compute pipeline created");
    return pipeline;
}

Result<std::unique_ptr<Pipeline>> Pipeline::createGraphics(const Device& device,
                                                           const GraphicsPipelineDesc& desc) {
    if (!desc.vertexShader || !desc.fragmentShader) {
        return Error{"Graphics pipeline needs a vertex and a fragment shader"};
    }

    // No descriptor sets yet (bindless textures land later); per-frame data
    // travels through push constants.
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pushRange.size = desc.pushConstantBytes;

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    if (desc.pushConstantBytes > 0) {
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &pushRange;
    }
    if (desc.descriptorLayout != nullptr) {
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &desc.descriptorLayout;
    }

    VkPipelineLayout layout = VK_NULL_HANDLE;
    if (VkResult r = vkCreatePipelineLayout(device.handle(), &layoutInfo, nullptr, &layout);
        r != VK_SUCCESS) {
        return Error{std::format("vkCreatePipelineLayout failed ({})", static_cast<int>(r))};
    }

    std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = desc.vertexShader->handle();
    stages[0].pName = desc.vertexEntryPoint;
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = desc.fragmentShader->handle();
    stages[1].pName = desc.fragmentEntryPoint;

    VkVertexInputBindingDescription binding{};
    binding.binding = 0;
    binding.stride = desc.vertexStride;
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    std::vector<VkVertexInputAttributeDescription> attributes;
    attributes.reserve(desc.vertexAttributes.size());
    for (const VertexAttribute& a : desc.vertexAttributes) {
        attributes.push_back({.location = a.location,
                              .binding = 0,
                              .format = static_cast<VkFormat>(a.format),
                              .offset = a.offset});
    }

    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    if (desc.vertexStride > 0) {
        vertexInput.vertexBindingDescriptionCount = 1;
        vertexInput.pVertexBindingDescriptions = &binding;
        vertexInput.vertexAttributeDescriptionCount =
            static_cast<std::uint32_t>(attributes.size());
        vertexInput.pVertexAttributeDescriptions = attributes.data();
    }

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewport{};
    viewport.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport.viewportCount = 1;
    viewport.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo raster{};
    raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisample{};
    multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // MRT (colorFormats non-empty) wins over the single colorFormat;
    // blend/write-mask state is identical across attachments.
    std::vector<VkFormat> colorFormats;
    if (!desc.colorFormats.empty()) {
        for (std::uint32_t format : desc.colorFormats) {
            colorFormats.push_back(static_cast<VkFormat>(format));
        }
    } else if (desc.colorFormat != 0) {
        colorFormats.push_back(static_cast<VkFormat>(desc.colorFormat));
    }

    VkPipelineColorBlendAttachmentState blendAttachment{};
    blendAttachment.blendEnable = (desc.alphaBlend || desc.occlusionDebug) ? VK_TRUE : VK_FALSE;
    blendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    blendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
    blendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    blendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
    blendAttachment.colorWriteMask =
        desc.occlusionProxy ? 0
                            : VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    const std::vector<VkPipelineColorBlendAttachmentState> blendAttachments(
        colorFormats.empty() ? 1 : colorFormats.size(), blendAttachment);

    VkPipelineColorBlendStateCreateInfo blend{};
    blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend.attachmentCount = static_cast<std::uint32_t>(blendAttachments.size());
    blend.pAttachments = blendAttachments.data();

    const std::array<VkDynamicState, 2> dynamicStates{VK_DYNAMIC_STATE_VIEWPORT,
                                                      VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{};
    dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic.dynamicStateCount = static_cast<std::uint32_t>(dynamicStates.size());
    dynamic.pDynamicStates = dynamicStates.data();

    // Y-flip is baked into the projection matrix (math::perspective), which
    // reverses winding — culling stays off until the winding convention is
    // settled alongside the pipeline XML.
    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = desc.disableDepthTest ? VK_FALSE : VK_TRUE;
    // Blended surfaces and occlusion proxies test against the opaque
    // depth but never write it.
    depthStencil.depthWriteEnable =
        (desc.alphaBlend || desc.occlusionProxy || desc.occlusionDebug || desc.background ||
         desc.disableDepthTest)
            ? VK_FALSE
            : VK_TRUE;
    // LESS_OR_EQUAL for proxies: a flat object's zero-extent box is
    // coplanar with its own rendered surface and must still pass.
    depthStencil.depthCompareOp =
        (desc.occlusionProxy || desc.occlusionDebug || desc.background)
            ? VK_COMPARE_OP_LESS_OR_EQUAL
            : VK_COMPARE_OP_LESS;

    // No color formats at all = depth-only pipeline (shadow passes): no
    // color attachment, no blend state.
    VkPipelineRenderingCreateInfo rendering{};
    rendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    if (!colorFormats.empty()) {
        rendering.colorAttachmentCount = static_cast<std::uint32_t>(colorFormats.size());
        rendering.pColorAttachmentFormats = colorFormats.data();
    } else {
        blend.attachmentCount = 0;
        blend.pAttachments = nullptr;
    }
    rendering.depthAttachmentFormat = static_cast<VkFormat>(desc.depthFormat);

    VkGraphicsPipelineCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    info.pNext = &rendering; // dynamic rendering: no VkRenderPass
    info.stageCount = static_cast<std::uint32_t>(stages.size());
    info.pStages = stages.data();
    info.pVertexInputState = &vertexInput;
    info.pInputAssemblyState = &inputAssembly;
    info.pViewportState = &viewport;
    info.pRasterizationState = &raster;
    info.pMultisampleState = &multisample;
    if (desc.depthFormat != 0) {
        info.pDepthStencilState = &depthStencil;
    }
    info.pColorBlendState = &blend;
    info.pDynamicState = &dynamic;
    info.layout = layout;
    info.renderPass = VK_NULL_HANDLE;

    VkPipeline handle = VK_NULL_HANDLE;
    if (VkResult r =
            vkCreateGraphicsPipelines(device.handle(), VK_NULL_HANDLE, 1, &info, nullptr, &handle);
        r != VK_SUCCESS) {
        vkDestroyPipelineLayout(device.handle(), layout, nullptr);
        return Error{std::format("vkCreateGraphicsPipelines failed ({})", static_cast<int>(r))};
    }

    auto pipeline = std::unique_ptr<Pipeline>(new Pipeline());
    pipeline->device_ = &device;
    pipeline->layout_ = layout;
    pipeline->pipeline_ = handle;
    log::info("Graphics pipeline created (dynamic rendering, {} color attachment(s), format {})",
              colorFormats.size(),
              colorFormats.empty() ? 0 : static_cast<int>(colorFormats.front()));
    return pipeline;
}

Pipeline::~Pipeline() {
    if (!device_) {
        return;
    }
    if (pipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_->handle(), pipeline_, nullptr);
    }
    if (layout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device_->handle(), layout_, nullptr);
    }
}

} // namespace rend::gpu
