use std::{mem::size_of, ops::Range};

use crate::{
    PickingLayerInstanceId, RenderContext, Rgba32Unmul,
    debug_label::DebugLabel,
    wgpu_resources::{BufferDesc, GpuBuffer},
};
use anyhow::Context;
use glam::Vec3;

/// Defines how mesh vertices are built.
pub mod mesh_vertices {
    use wgpu::VertexStepMode;

    use crate::wgpu_resources::VertexBufferLayout;

    /// Vertex buffer layouts describing how vertex data should be laid out.
    pub fn vertex_buffer_layouts_color() -> smallvec::SmallVec<[VertexBufferLayout; 4]> {
        let mut layouts = VertexBufferLayout::from_formats(
            [
                wgpu::VertexFormat::Float32x3, // position
                wgpu::VertexFormat::Unorm8x4,  // RGBA
            ]
            .into_iter(),
        );

        // FIXME: clean this up..
        layouts
            .iter_mut()
            .for_each(|l| l.step_mode = VertexStepMode::Instance);

        layouts
    }

    pub fn vertex_buffer_layouts_data() -> smallvec::SmallVec<[VertexBufferLayout; 4]> {
        let mut layouts = VertexBufferLayout::from_formats(
            [
                wgpu::VertexFormat::Float32x3, // position
                wgpu::VertexFormat::Unorm8x4,  // RGBA
                wgpu::VertexFormat::Uint32x2,  // picking layer instance id / outline mask
            ]
            .into_iter(),
        );

        // FIXME: clean this up..
        layouts
            .iter_mut()
            .for_each(|l| l.step_mode = VertexStepMode::Instance);

        layouts
    }
}

pub mod gpu_data {
    use crate::{draw_phases::PickingLayerObjectId, wgpu_buffer_types};

    /// Uniform buffer that changes for every batch of points.
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    pub struct UniformBuffer {
        pub world_from_obj: wgpu_buffer_types::Mat4,

        pub picking_object_id: PickingLayerObjectId,
        pub outline_mask_ids: wgpu_buffer_types::UVec2, // currently ignored.. just here for padding

        pub point_size: wgpu_buffer_types::Vec4, // only the first float is actually used, the rest is padding

        pub end_padding: [wgpu_buffer_types::PaddingRow; 16 - 6],
    }
}

#[derive(Clone)]
pub struct GPUPersistentPointCloud {
    /// Buffer for all vertex data, subdivided in several sections for different vertex buffer bindings.
    /// See [`mesh_vertices`]
    pub point_count: u64,
    pub point_buffer_combined: GpuBuffer,
    pub point_buffer_positions_range: Range<u64>,
    pub point_buffer_colors_range: Range<u64>,
    pub point_buffer_picking_range: Option<Range<u64>>,
    pub point_buffer_outline_range: Option<Range<u64>>,
}

pub struct CPUPointCloud<'t> {
    pub label: DebugLabel,
    pub positions: &'t [glam::Vec3],
    pub colors: &'t [Rgba32Unmul],
    pub picking: Option<&'t [PickingLayerInstanceId]>,
    pub outline: Option<&'t [glam::UVec2]>, // FIXME: overkill af
}

impl GPUPersistentPointCloud {
    // TODO(andreas): Take read-only context here and make uploads happen on staging belt.
    pub fn new(ctx: &RenderContext, data: CPUPointCloud<'_>) -> anyhow::Result<Self> {
        re_tracing::profile_function!();

        // TODO(andreas): Have a variant that gets this from a stack allocator.
        let point_count = data.positions.len();
        let vb_positions_size = (point_count * size_of::<glam::Vec3>()) as u64;
        let vb_color_size = (point_count * size_of::<Rgba32Unmul>()) as u64;
        let vb_picking_size = (point_count * size_of::<glam::UVec2>()) as u64;
        let vb_outline_size = (point_count * size_of::<glam::UVec2>()) as u64;

        let vb_combined_size = {
            vb_positions_size
                + vb_color_size
                + if data.picking.is_some() {
                    vb_picking_size
                } else {
                    0u64
                }
                + if data.outline.is_some() {
                    vb_outline_size
                } else {
                    0u64
                }
        };

        let pools = &ctx.gpu_resources;
        let device = &ctx.device;

        let point_buffer_combined = pools.buffers.alloc(
            device,
            &BufferDesc {
                label: format!("{} - vertices", data.label).into(),
                size: vb_combined_size,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        );

        let mut staging_buffer = ctx.cpu_write_gpu_read_belt.lock().allocate::<u8>(
            &ctx.device,
            &ctx.gpu_resources.buffers,
            vb_combined_size as _,
        )?;

        staging_buffer.extend_from_slice(bytemuck::cast_slice(data.positions))?;
        let point_buffer_positions_range = 0..vb_positions_size;

        staging_buffer.extend_from_slice(bytemuck::cast_slice(data.colors))?;
        let point_buffer_colors_range = vb_positions_size..(vb_positions_size + vb_color_size);

        let point_buffer_picking_range = if let Some(picking) = data.picking {
            staging_buffer.extend_from_slice(bytemuck::cast_slice(picking))?;
            Some(point_buffer_colors_range.end..(point_buffer_colors_range.end + vb_picking_size))
        } else {
            None
        };
        let point_buffer_outline_range = if let Some(outline) = data.outline {
            staging_buffer.extend_from_slice(bytemuck::cast_slice(outline))?;
            let from = point_buffer_picking_range
                .clone()
                .unwrap_or(point_buffer_colors_range.clone())
                .end;
            Some(from..(from + vb_outline_size))
        } else {
            None
        };
        staging_buffer.copy_to_buffer(
            ctx.active_frame.before_view_builder_encoder.lock().get(),
            &point_buffer_combined,
            0,
        )?;

        Ok(Self {
            point_count: point_count as u64,
            point_buffer_combined,
            point_buffer_positions_range,
            point_buffer_colors_range,
            point_buffer_picking_range,
            point_buffer_outline_range,
        })
    }

    pub fn update_outline(
        &self,
        ctx: &RenderContext,
        outline: &[glam::UVec2],
    ) -> anyhow::Result<()> {
        let outline_range = self
            .point_buffer_outline_range
            .clone()
            .context("Cloud has no outline buffer")?;
        let size = outline_range.end - outline_range.start;
        let mut staging_buffer = ctx.cpu_write_gpu_read_belt.lock().allocate::<u8>(
            &ctx.device,
            &ctx.gpu_resources.buffers,
            size as _,
        )?;
        staging_buffer.extend_from_slice(bytemuck::cast_slice(outline))?;
        staging_buffer.copy_to_buffer(
            ctx.active_frame.before_view_builder_encoder.lock().get(),
            &self.point_buffer_combined,
            outline_range.start,
        )?;
        anyhow::Ok(())
    }

    pub fn update_color(&self, ctx: &RenderContext, colors: &[Rgba32Unmul]) -> anyhow::Result<()> {
        let size = self.point_buffer_colors_range.end - self.point_buffer_colors_range.start;
        let mut staging_buffer = ctx.cpu_write_gpu_read_belt.lock().allocate::<u8>(
            &ctx.device,
            &ctx.gpu_resources.buffers,
            size as _,
        )?;
        staging_buffer.extend_from_slice(bytemuck::cast_slice(colors))?;
        staging_buffer.copy_to_buffer(
            ctx.active_frame.before_view_builder_encoder.lock().get(),
            &self.point_buffer_combined,
            self.point_buffer_colors_range.start,
        )?;
        anyhow::Ok(())
    }

    pub fn update_positions(&self, ctx: &RenderContext, positions: &[Vec3]) -> anyhow::Result<()> {
        let size = self.point_buffer_positions_range.end - self.point_buffer_positions_range.start;
        let mut staging_buffer = ctx.cpu_write_gpu_read_belt.lock().allocate::<u8>(
            &ctx.device,
            &ctx.gpu_resources.buffers,
            size as _,
        )?;
        staging_buffer.extend_from_slice(bytemuck::cast_slice(positions))?;
        staging_buffer.copy_to_buffer(
            ctx.active_frame.before_view_builder_encoder.lock().get(),
            &self.point_buffer_combined,
            self.point_buffer_positions_range.start,
        )?;
        anyhow::Ok(())
    }
}
