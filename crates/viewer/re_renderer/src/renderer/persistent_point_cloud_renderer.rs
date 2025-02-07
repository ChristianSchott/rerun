//! Persistent Point-Mesh renderer.

use std::{num::NonZeroU64, sync::Arc};

use enumset::{EnumSet, enum_set};
use glam::UVec2;
use smallvec::smallvec;

use crate::{
    PickingLayerObjectId, PickingLayerProcessor,
    allocator::create_and_fill_uniform_buffer,
    draw_phases::{DrawPhase, OutlineMaskProcessor},
    include_shader_module,
    persistent_point_cloud::{self, GPUPersistentPointCloud, gpu_data, mesh_vertices},
    renderer::{DrawDataDrawable, DrawInstruction},
    view_builder::ViewBuilder,
    wgpu_resources::{
        BindGroupDesc, BindGroupLayoutDesc, GpuBindGroup, GpuBindGroupLayoutHandle,
        GpuRenderPipelineHandle, GpuRenderPipelinePoolAccessor, PipelineLayoutDesc,
        RenderPipelineDesc,
    },
};

use super::{DrawData, DrawError, RenderContext, Renderer};

#[derive(Clone)]
pub struct PersistentPointCloudDrawData {
    point_cloud: Arc<GPUPersistentPointCloud>,
    point_cloud_bind_group: GpuBindGroup,
    active_phases: EnumSet<DrawPhase>,
}

impl DrawData for PersistentPointCloudDrawData {
    type Renderer = PersistentPointCloudRenderer;

    fn collect_drawables(
        &self,
        _view_info: &super::DrawableCollectionViewInfo,
        collector: &mut crate::DrawableCollector<'_>,
    ) {
        collector.add_drawable(
            self.active_phases,
            DrawDataDrawable {
                // TODO(andreas): Don't have distance information yet. For now just always draw points last since they're quite expensive.
                distance_sort_key: f32::MAX,
                draw_data_payload: 0u32,
            },
        );
    }
}

impl PersistentPointCloudDrawData {
    pub fn new(
        ctx: &RenderContext,
        point_cloud: Arc<GPUPersistentPointCloud>,
        world_from_point_cloud: glam::Affine3A,
        picking_object_id: PickingLayerObjectId,
        enable_outline: bool,
        point_size: f32,
    ) -> Self {
        re_tracing::profile_function!();

        let point_cloud_bind_group = {
            let point_renderer = ctx.renderer::<PersistentPointCloudRenderer>();
            let all_points_buffer_binding = create_and_fill_uniform_buffer(
                ctx,
                "PersistentPointCloud::DrawDataUniformBuffer".into(),
                gpu_data::UniformBuffer {
                    world_from_obj: world_from_point_cloud.into(),
                    picking_object_id,
                    outline_mask_ids: UVec2::default().into(),
                    point_size: glam::Vec4::new(point_size, 0f32, 0f32, 0f32).into(),
                    end_padding: Default::default(),
                },
            );

            ctx.gpu_resources.bind_groups.alloc(
                &ctx.device,
                &ctx.gpu_resources,
                &BindGroupDesc {
                    label: "PersistentPointCloudDrawData::bind_group_all_points".into(),
                    entries: smallvec![all_points_buffer_binding],
                    layout: point_renderer.bind_group_layout_all_points,
                },
            )
        };

        let active_phases = {
            let mut active_phases = enum_set![DrawPhase::Opaque];
            if picking_object_id.0 != 0 && point_cloud.point_buffer_picking_range.is_some() {
                active_phases.insert(DrawPhase::PickingLayer);
            }
            if enable_outline && point_cloud.point_buffer_outline_range.is_some() {
                active_phases.insert(DrawPhase::OutlineMask);
            }
            active_phases
        };

        Self {
            point_cloud,
            point_cloud_bind_group,
            active_phases,
        }
    }
}

pub struct PersistentPointCloudRenderer {
    render_pipeline_shaded: GpuRenderPipelineHandle,
    render_pipeline_picking_layer: GpuRenderPipelineHandle,
    render_pipeline_outline_mask: GpuRenderPipelineHandle,
    bind_group_layout_all_points: GpuBindGroupLayoutHandle,
}

impl Renderer for PersistentPointCloudRenderer {
    type RendererDrawData = PersistentPointCloudDrawData;

    fn create_renderer(ctx: &RenderContext) -> Self {
        re_tracing::profile_function!();

        let render_pipelines = &ctx.gpu_resources.render_pipelines;

        let bind_group_layout_all_points = ctx.gpu_resources.bind_group_layouts.get_or_create(
            &ctx.device,
            &BindGroupLayoutDesc {
                label: "PersistentPointCloudRenderer::bind_group_layout_all_points".into(),
                entries: vec![wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
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
            topology: wgpu::PrimitiveTopology::TriangleList, // FIXME: just use wgpu::PrimitiveState::default()
            ..Default::default()
        };

        // TODO: make this customizable
        // let color_target = wgpu::ColorTargetState {
        //     format: ViewBuilder::MAIN_TARGET_COLOR_FORMAT,
        //     blend: Some(BlendState::ALPHA_BLENDING),
        //     write_mask: ColorWrites::ALL,
        // };

        let render_pipeline_shaded_desc = RenderPipelineDesc {
            label: "PersistentPointCloudRenderer::render_pipeline_shaded".into(),
            pipeline_layout,
            vertex_entrypoint: "vs_main_color".into(),
            vertex_handle: shader_module,
            fragment_entrypoint: "fs_main_shaded".into(),
            fragment_handle: shader_module,
            vertex_buffers: mesh_vertices::vertex_buffer_layouts_color(),
            render_targets: smallvec![Some(ViewBuilder::MAIN_TARGET_ALPHA_TO_COVERAGE_COLOR_STATE)], //Some(color_target)],
            primitive,
            depth_stencil: Some(ViewBuilder::MAIN_TARGET_DEFAULT_DEPTH_STATE),
            multisample: ViewBuilder::main_target_default_msaa_state(ctx.render_config(), false),
        };
        let render_pipeline_shaded =
            render_pipelines.get_or_create(ctx, &render_pipeline_shaded_desc);
        let render_pipeline_picking_layer = render_pipelines.get_or_create(
            ctx,
            &RenderPipelineDesc {
                label: "PersistentPointCloudRenderer::render_pipeline_picking_layer".into(),
                fragment_entrypoint: "fs_main_picking_layer".into(),
                vertex_entrypoint: "vs_main_data".into(),
                vertex_buffers: mesh_vertices::vertex_buffer_layouts_data(),
                render_targets: smallvec![Some(PickingLayerProcessor::PICKING_LAYER_FORMAT.into())],
                depth_stencil: PickingLayerProcessor::PICKING_LAYER_DEPTH_STATE,
                multisample: PickingLayerProcessor::PICKING_LAYER_MSAA_STATE,
                ..render_pipeline_shaded_desc.clone()
            },
        );
        let render_pipeline_outline_mask = render_pipelines.get_or_create(
            ctx,
            &RenderPipelineDesc {
                label: "PersistentPointCloudRenderer::render_pipeline_outline_mask".into(),
                fragment_entrypoint: "fs_main_outline_mask".into(),
                vertex_entrypoint: "vs_main_data".into(),
                vertex_buffers: mesh_vertices::vertex_buffer_layouts_data(),
                render_targets: smallvec![Some(OutlineMaskProcessor::MASK_FORMAT.into())],
                depth_stencil: OutlineMaskProcessor::MASK_DEPTH_STATE,
                multisample: OutlineMaskProcessor::mask_default_msaa_state(ctx.device_caps().tier),
                ..render_pipeline_shaded_desc
            },
        );

        Self {
            render_pipeline_shaded,
            bind_group_layout_all_points,
            render_pipeline_picking_layer,
            render_pipeline_outline_mask,
        }
    }

    fn draw(
        &self,
        render_pipelines: &GpuRenderPipelinePoolAccessor<'_>,
        phase: DrawPhase,
        pass: &mut wgpu::RenderPass<'_>,
        draw_instructions: &[DrawInstruction<'_, Self::RendererDrawData>],
    ) -> Result<(), DrawError> {
        re_tracing::profile_function!();

        let pipeline_handle = match phase {
            DrawPhase::Opaque => self.render_pipeline_shaded,
            DrawPhase::OutlineMask => self.render_pipeline_outline_mask,
            DrawPhase::PickingLayer => self.render_pipeline_picking_layer,
            _ => unreachable!("We were called on a phase we weren't subscribed to: {phase:?}"),
        };
        let pipeline = render_pipelines.get(pipeline_handle)?;

        pass.set_pipeline(pipeline);

        for DrawInstruction { draw_data, .. } in draw_instructions {
            if !draw_data.active_phases.contains(phase) {
                continue;
            }
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
            match phase {
                DrawPhase::PickingLayer => {
                    if let Some(range) = draw_data.point_cloud.point_buffer_picking_range.clone() {
                        pass.set_vertex_buffer(
                            2,
                            draw_data.point_cloud.point_buffer_combined.slice(range),
                        );
                    }
                }
                DrawPhase::OutlineMask => {
                    if let Some(range) = draw_data.point_cloud.point_buffer_outline_range.clone() {
                        pass.set_vertex_buffer(
                            2,
                            draw_data.point_cloud.point_buffer_combined.slice(range),
                        );
                    }
                }
                _ => (),
            }

            pass.draw(0..3, 0..draw_data.point_cloud.point_count as u32);
            // pass.draw(0..draw_data.point_cloud.point_count as u32, 0..1);
        }

        Ok(())
    }
}
