//! Static Point-Mesh renderer.

use std::{num::NonZeroU64, sync::Arc};

use smallvec::smallvec;

use crate::{
    allocator::create_and_fill_uniform_buffer,
    draw_phases::DrawPhase,
    include_shader_module,
    persistent_point_cloud::{self, gpu_data, mesh_vertices, GPUPersistentPointCloud},
    view_builder::ViewBuilder,
    wgpu_resources::{
        BindGroupDesc, BindGroupLayoutDesc, GpuBindGroup, GpuBindGroupLayoutHandle,
        GpuRenderPipelineHandle, GpuRenderPipelinePoolAccessor, PipelineLayoutDesc,
        RenderPipelineDesc,
    },
    CpuWriteGpuReadError, OutlineMaskPreference, PickingLayerObjectId, PickingLayerProcessor,
};

use super::{DrawData, DrawError, RenderContext, Renderer};

#[derive(Clone)]
pub struct PersistentPointCloudDrawData {
    point_cloud: Arc<GPUPersistentPointCloud>,
    point_cloud_bind_group: GpuBindGroup,
}

impl DrawData for PersistentPointCloudDrawData {
    type Renderer = PersistentPointCloudRenderer;
}

impl PersistentPointCloudDrawData {
    /// Transforms and uploads mesh instance data to be consumed by gpu.
    ///
    /// Try bundling all mesh instances into a single draw data instance whenever possible.
    /// If you pass zero mesh instances, subsequent drawing will do nothing.
    /// Mesh data itself is gpu uploaded if not already present.
    pub fn new(
        ctx: &RenderContext,
        point_cloud: Arc<GPUPersistentPointCloud>,
        world_from_point_cloud: glam::Affine3A,
        picking_object_id: PickingLayerObjectId,
        outline: OutlineMaskPreference,
    ) -> Result<Self, CpuWriteGpuReadError> {
        re_tracing::profile_function!();

        let point_cloud_bind_group = {
            let point_renderer = ctx.renderer::<PersistentPointCloudRenderer>();
            let all_points_buffer_binding = create_and_fill_uniform_buffer(
                ctx,
                "PersistentPointCloud::DrawDataUniformBuffer".into(),
                gpu_data::UniformBuffer {
                    world_from_obj: world_from_point_cloud.into(),
                    outline_mask_ids: outline.0.unwrap_or_default().into(),
                    picking_object_id,
                    end_padding: Default::default(),
                },
            );

            ctx.gpu_resources.bind_groups.alloc(
                &ctx.device,
                &ctx.gpu_resources,
                &BindGroupDesc {
                    label: "PersistentPointCloudDrawData::bind_group_all_points".into(),
                    entries: smallvec![all_points_buffer_binding,],
                    layout: point_renderer.bind_group_layout_all_points,
                },
            )
        };

        Ok(Self {
            point_cloud,
            point_cloud_bind_group,
        })
    }
}

pub struct PersistentPointCloudRenderer {
    render_pipeline_shaded: GpuRenderPipelineHandle,
    render_pipeline_picking_layer: GpuRenderPipelineHandle,
    // render_pipeline_outline_mask: GpuRenderPipelineHandle,
    bind_group_layout_all_points: GpuBindGroupLayoutHandle,
}

impl Renderer for PersistentPointCloudRenderer {
    type RendererDrawData = PersistentPointCloudDrawData;

    fn participated_phases() -> &'static [DrawPhase] {
        &[
            DrawPhase::Opaque,
            // DrawPhase::OutlineMask,
            DrawPhase::PickingLayer,
        ]
    }

    fn create_renderer(ctx: &RenderContext) -> Self {
        re_tracing::profile_function!();

        let render_pipelines = &ctx.gpu_resources.render_pipelines;

        let bind_group_layout_all_points = ctx.gpu_resources.bind_group_layouts.get_or_create(
            &ctx.device,
            &BindGroupLayoutDesc {
                label: "PersistentPointCloudRenderer::bind_group_layout_all_points".into(),
                entries: vec![wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(std::mem::size_of::<
                            persistent_point_cloud::gpu_data::UniformBuffer,
                        >() as _),
                    },
                    count: None,
                }],
            },
        );

        let pipeline_layout = ctx.gpu_resources.pipeline_layouts.get_or_create(
            ctx,
            &PipelineLayoutDesc {
                label: "StaticPointCloudRenderer::pipeline_layout".into(),
                entries: vec![ctx.global_bindings.layout, bind_group_layout_all_points],
            },
        );

        let shader_module = ctx.gpu_resources.shader_modules.get_or_create(
            ctx,
            &include_shader_module!("../../shader/static_point_cloud.wgsl"),
        );

        let primitive = wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::PointList,
            ..Default::default()
        };

        let render_pipeline_shaded_desc = RenderPipelineDesc {
            label: "PersistentPointCloudRenderer::render_pipeline_shaded".into(),
            pipeline_layout,
            vertex_entrypoint: "vs_main".into(),
            vertex_handle: shader_module,
            fragment_entrypoint: "fs_main_shaded".into(),
            fragment_handle: shader_module,
            vertex_buffers: mesh_vertices::vertex_buffer_layouts(),
            render_targets: smallvec![Some(ViewBuilder::MAIN_TARGET_COLOR_FORMAT.into())],
            primitive,
            depth_stencil: ViewBuilder::MAIN_TARGET_DEFAULT_DEPTH_STATE,
            multisample: ViewBuilder::MAIN_TARGET_DEFAULT_MSAA_STATE,
        };
        let render_pipeline_shaded =
            render_pipelines.get_or_create(ctx, &render_pipeline_shaded_desc);
        let render_pipeline_picking_layer = render_pipelines.get_or_create(
            ctx,
            &RenderPipelineDesc {
                label: "PersistentPointCloudRenderer::render_pipeline_picking_layer".into(),
                fragment_entrypoint: "fs_main_picking_layer".into(),
                render_targets: smallvec![Some(PickingLayerProcessor::PICKING_LAYER_FORMAT.into())],
                depth_stencil: PickingLayerProcessor::PICKING_LAYER_DEPTH_STATE,
                multisample: PickingLayerProcessor::PICKING_LAYER_MSAA_STATE,
                ..render_pipeline_shaded_desc.clone()
            },
        );
        // let render_pipeline_outline_mask = render_pipelines.get_or_create(
        //     ctx,
        //     &RenderPipelineDesc {
        //         label: "StaticPointCloudRenderer::render_pipeline_outline_mask".into(),
        //         fragment_entrypoint: "fs_main_outline_mask".into(),
        //         render_targets: smallvec![Some(OutlineMaskProcessor::MASK_FORMAT.into())],
        //         depth_stencil: OutlineMaskProcessor::MASK_DEPTH_STATE,
        //         multisample: OutlineMaskProcessor::mask_default_msaa_state(ctx.device_caps().tier),
        //         ..render_pipeline_shaded_desc
        //     },
        // );

        Self {
            render_pipeline_shaded,
            bind_group_layout_all_points,
            render_pipeline_picking_layer,
            // render_pipeline_outline_mask,
        }
    }

    fn draw(
        &self,
        render_pipelines: &GpuRenderPipelinePoolAccessor<'_>,
        phase: DrawPhase,
        pass: &mut wgpu::RenderPass<'_>,
        draw_data: &Self::RendererDrawData,
    ) -> Result<(), DrawError> {
        re_tracing::profile_function!();

        let pipeline_handle = match phase {
            DrawPhase::Opaque => self.render_pipeline_shaded,
            // DrawPhase::OutlineMask => self.render_pipeline_outline_mask,
            DrawPhase::PickingLayer => self.render_pipeline_picking_layer,
            _ => unreachable!("We were called on a phase we weren't subscribed to: {phase:?}"),
        };
        let pipeline = render_pipelines.get(pipeline_handle)?;

        pass.set_pipeline(pipeline);
        pass.set_bind_group(1, &draw_data.point_cloud_bind_group, &[]);

        pass.set_vertex_buffer(
            0,
            draw_data
                .point_cloud
                .point_buffer_combined
                .slice(draw_data.point_cloud.point_buffer_positions_range.clone()),
        );
        pass.set_vertex_buffer(
            1,
            draw_data
                .point_cloud
                .point_buffer_combined
                .slice(draw_data.point_cloud.point_buffer_colors_range.clone()),
        );
        pass.set_vertex_buffer(
            2,
            draw_data
                .point_cloud
                .point_buffer_combined
                .slice(draw_data.point_cloud.point_buffer_picking_range.clone()),
        );

        pass.draw(0..draw_data.point_cloud.point_count as u32, 0..1);

        Ok(())
    }
}
