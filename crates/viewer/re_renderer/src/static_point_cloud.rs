use std::{mem::size_of, ops::Range};

use crate::{
    debug_label::DebugLabel,
    wgpu_resources::{BufferDesc, GpuBuffer},
    RenderContext, Rgba32Unmul,
};

/// Defines how mesh vertices are built.
pub mod mesh_vertices {
    use crate::wgpu_resources::VertexBufferLayout;

    /// Vertex buffer layouts describing how vertex data should be laid out.
    ///
    /// Needs to be kept in sync with `mesh_vertex.wgsl`.
    pub fn vertex_buffer_layouts() -> smallvec::SmallVec<[VertexBufferLayout; 4]> {
        // TODO(andreas): Compress normals. Afaik Octahedral Mapping is the best by far, see https://jcgt.org/published/0003/02/01/
        VertexBufferLayout::from_formats(
            [
                wgpu::VertexFormat::Float32x3, // position
                wgpu::VertexFormat::Unorm8x4,  // RGBA
            ]
            .into_iter(),
        )
    }

    /// Next vertex attribute index that can be used for another vertex buffer.
    pub fn next_free_shader_location() -> u32 {
        vertex_buffer_layouts()
            .iter()
            .flat_map(|layout| layout.attributes.iter())
            .max_by(|a1, a2| a1.shader_location.cmp(&a2.shader_location))
            .unwrap()
            .shader_location
            + 1
    }
}

#[derive(Clone)]
pub struct GPUStaticPointCloud {
    /// Buffer for all vertex data, subdivided in several sections for different vertex buffer bindings.
    /// See [`mesh_vertices`]
    pub point_count: u64,
    pub vertex_buffer_combined: GpuBuffer,
    pub vertex_buffer_positions_range: Range<u64>,
    pub vertex_buffer_colors_range: Range<u64>,
    // pub vertex_buffer_normals_range: Range<u64>,
}

pub struct CPUPointCloud<'t> {
    pub label: DebugLabel,
    pub positions: &'t [glam::Vec3],
    pub colors: &'t [Rgba32Unmul],
}

impl GPUStaticPointCloud {
    // TODO(andreas): Take read-only context here and make uploads happen on staging belt.
    pub fn new(ctx: &RenderContext, data: CPUPointCloud<'_>) -> anyhow::Result<Self> {
        re_tracing::profile_function!();

        re_log::trace!(
            "uploading new mesh named {:?} with {} points",
            data.label.get(),
            data.positions.len(),
        );

        // TODO(andreas): Have a variant that gets this from a stack allocator.
        let vb_positions_size = (data.positions.len() * size_of::<glam::Vec3>()) as u64;
        let vb_color_size = (data.colors.len() * size_of::<Rgba32Unmul>()) as u64;

        let vb_combined_size = vb_positions_size + vb_color_size;

        let pools = &ctx.gpu_resources;
        let device = &ctx.device;

        let vertex_buffer_combined = {
            let vertex_buffer_combined = pools.buffers.alloc(
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
            staging_buffer.extend_from_slice(bytemuck::cast_slice(&data.positions))?;
            staging_buffer.extend_from_slice(bytemuck::cast_slice(&data.colors))?;
            staging_buffer.copy_to_buffer(
                ctx.active_frame.before_view_builder_encoder.lock().get(),
                &vertex_buffer_combined,
                0,
            )?;
            vertex_buffer_combined
        };

        Ok(Self {
            vertex_buffer_combined,
            point_count: data.positions.len() as u64,
            vertex_buffer_positions_range: 0..vb_positions_size,
            vertex_buffer_colors_range: vb_positions_size..vb_combined_size,
        })
    }
}
