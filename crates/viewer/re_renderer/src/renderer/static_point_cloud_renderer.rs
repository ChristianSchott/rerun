//! Static Point-Mesh renderer.

use std::sync::Arc;

use smallvec::smallvec;

use crate::{
    draw_phases::DrawPhase,
    include_shader_module,
    static_point_cloud::{mesh_vertices, GPUStaticPointCloud},
    view_builder::ViewBuilder,
    wgpu_resources::{
        GpuRenderPipelineHandle, GpuRenderPipelinePoolAccessor, PipelineLayoutDesc,
        RenderPipelineDesc,
    },
    CpuWriteGpuReadError,
};

use super::{DrawData, DrawError, RenderContext, Renderer};

#[derive(Clone)]
pub struct StaticPointCloudDrawData {
    point_cloud: Arc<GPUStaticPointCloud>,
    world_from_point_cloud: glam::Affine3A,
}

impl DrawData for StaticPointCloudDrawData {
    type Renderer = StaticPointCloudRenderer;
}

impl StaticPointCloudDrawData {
    /// Transforms and uploads mesh instance data to be consumed by gpu.
    ///
    /// Try bundling all mesh instances into a single draw data instance whenever possible.
    /// If you pass zero mesh instances, subsequent drawing will do nothing.
    /// Mesh data itself is gpu uploaded if not already present.
    pub fn new(
        ctx: &RenderContext,
        pc: Arc<GPUStaticPointCloud>,
        world_from_point_cloud: glam::Affine3A,
    ) -> Result<Self, CpuWriteGpuReadError> {
        re_tracing::profile_function!();

        let _pc_renderer = ctx.renderer::<StaticPointCloudRenderer>();

        Ok(Self {
            point_cloud: pc,
            world_from_point_cloud,
        })
    }
}

pub struct StaticPointCloudRenderer {
    render_pipeline_shaded: GpuRenderPipelineHandle,
    // render_pipeline_picking_layer: GpuRenderPipelineHandle,
    // render_pipeline_outline_mask: GpuRenderPipelineHandle,
}

impl Renderer for StaticPointCloudRenderer {
    type RendererDrawData = StaticPointCloudDrawData;

    fn participated_phases() -> &'static [DrawPhase] {
        &[
            DrawPhase::Opaque,
            // DrawPhase::OutlineMask,
            // DrawPhase::PickingLayer,
        ]
    }

    fn create_renderer(ctx: &RenderContext) -> Self {
        re_tracing::profile_function!();

        let render_pipelines = &ctx.gpu_resources.render_pipelines;

        let pipeline_layout = ctx.gpu_resources.pipeline_layouts.get_or_create(
            ctx,
            &PipelineLayoutDesc {
                label: "StaticPointCloudRenderer::pipeline_layout".into(),
                entries: vec![ctx.global_bindings.layout],
            },
        );

        let shader_module = ctx.gpu_resources.shader_modules.get_or_create(
            ctx,
            &include_shader_module!("../../shader/static_point_cloud.wgsl"),
        );

        let primitive = wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::PointList,
            cull_mode: None, //Some(wgpu::Face::Back), // TODO(andreas): Need to specify from outside if mesh is CW or CCW?
            ..Default::default()
        };

        let render_pipeline_shaded_desc = RenderPipelineDesc {
            label: "StaticPointCloudRenderer::render_pipeline_shaded".into(),
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
        // let render_pipeline_picking_layer = render_pipelines.get_or_create(
        //     ctx,
        //     &RenderPipelineDesc {
        //         label: "StaticPointCloudRenderer::render_pipeline_picking_layer".into(),
        //         fragment_entrypoint: "fs_main_picking_layer".into(),
        //         render_targets: smallvec![Some(PickingLayerProcessor::PICKING_LAYER_FORMAT.into())],
        //         depth_stencil: PickingLayerProcessor::PICKING_LAYER_DEPTH_STATE,
        //         multisample: PickingLayerProcessor::PICKING_LAYER_MSAA_STATE,
        //         ..render_pipeline_shaded_desc.clone()
        //     },
        // );
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
            // render_pipeline_picking_layer,
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
            // DrawPhase::PickingLayer => self.render_pipeline_picking_layer,
            _ => unreachable!("We were called on a phase we weren't subscribed to: {phase:?}"),
        };
        let pipeline = render_pipelines.get(pipeline_handle)?;

        pass.set_pipeline(pipeline);

        pass.set_vertex_buffer(
            0,
            draw_data
                .point_cloud
                .vertex_buffer_combined
                .slice(draw_data.point_cloud.vertex_buffer_positions_range.clone()),
        );
        pass.set_vertex_buffer(
            1,
            draw_data
                .point_cloud
                .vertex_buffer_combined
                .slice(draw_data.point_cloud.vertex_buffer_colors_range.clone()),
        );

        // pass.set_bind_group(index, bind_group, offsets);
        pass.draw(0..draw_data.point_cloud.point_count as u32, 0..1);

        // let mut instance_start_index = 0;

        // for mesh_batch in &draw_data.batches {
        //     if phase == DrawPhase::OutlineMask && mesh_batch.count_with_outlines == 0 {
        //         instance_start_index += mesh_batch.count;
        //         continue;
        //     }

        //     let vertex_buffer_combined = &mesh_batch.mesh.vertex_buffer_combined;
        //     let index_buffer = &mesh_batch.mesh.index_buffer;

        //     pass.set_vertex_buffer(
        //         1,
        //         vertex_buffer_combined.slice(mesh_batch.mesh.vertex_buffer_positions_range.clone()),
        //     );
        //     pass.set_vertex_buffer(
        //         2,
        //         vertex_buffer_combined.slice(mesh_batch.mesh.vertex_buffer_colors_range.clone()),
        //     );
        //     pass.set_vertex_buffer(
        //         3,
        //         vertex_buffer_combined.slice(mesh_batch.mesh.vertex_buffer_normals_range.clone()),
        //     );
        //     pass.set_vertex_buffer(
        //         4,
        //         vertex_buffer_combined.slice(mesh_batch.mesh.vertex_buffer_texcoord_range.clone()),
        //     );
        //     pass.set_index_buffer(
        //         index_buffer.slice(mesh_batch.mesh.index_buffer_range.clone()),
        //         wgpu::IndexFormat::Uint32,
        //     );

        //     let num_meshes_to_draw = if phase == DrawPhase::OutlineMask {
        //         mesh_batch.count_with_outlines
        //     } else {
        //         mesh_batch.count
        //     };
        //     let instance_range = instance_start_index..(instance_start_index + num_meshes_to_draw);

        //     for material in &mesh_batch.mesh.materials {
        //         debug_assert!(num_meshes_to_draw > 0);

        //         pass.set_bind_group(1, &material.bind_group, &[]);
        //         pass.draw_indexed(material.index_range.clone(), 0, instance_range.clone());
        //     }

        //     // Advance instance start index with *total* number of instances in this batch.
        //     instance_start_index += mesh_batch.count;
        // }

        Ok(())
    }
}
