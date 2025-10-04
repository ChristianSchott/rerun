use crate::{RenderContext, debug_label::DebugLabel};

use super::{
    pipeline_layout_pool::{GpuPipelineLayoutHandle, GpuPipelineLayoutPool},
    resource::PoolError,
    shader_module_pool::{GpuShaderModuleHandle, GpuShaderModulePool},
    static_resource_pool::{StaticResourcePool, StaticResourcePoolReadLockAccessor},
};

slotmap::new_key_type! { pub struct GpuComputePipelineHandle; }

/// ComputePipeline descriptor, can be converted into [`wgpu::ComputePipeline`] (which isn't hashable or comparable)
#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct ComputePipelineDesc {
    /// Debug label of the pipeline. This will show up in graphics debuggers for easy identification.
    pub label: DebugLabel,

    pub pipeline_layout: GpuPipelineLayoutHandle,

    pub compute_entrypoint: String,
    pub compute_handle: GpuShaderModuleHandle,
}

#[derive(thiserror::Error, Debug)]
pub enum RenderPipelineCreationError {
    #[error("Referenced pipeline layout not found: {0}")]
    PipelineLayout(PoolError),

    #[error("Referenced compute shader not found: {0}")]
    ComputeShaderNotFound(PoolError),
}

impl ComputePipelineDesc {
    fn create_compute_pipeline(
        &self,
        device: &wgpu::Device,
        pipeline_layouts: &GpuPipelineLayoutPool,
        shader_modules: &GpuShaderModulePool,
    ) -> Result<wgpu::ComputePipeline, RenderPipelineCreationError> {
        let pipeline_layouts = pipeline_layouts.resources();
        let pipeline_layout = pipeline_layouts
            .get(self.pipeline_layout)
            .map_err(RenderPipelineCreationError::PipelineLayout)?;

        let shader_modules = shader_modules.resources();
        let compute_shader_module = shader_modules
            .get(self.compute_handle)
            .map_err(RenderPipelineCreationError::ComputeShaderNotFound)?;

        Ok(
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: self.label.get(),
                layout: Some(pipeline_layout),
                module: compute_shader_module,
                entry_point: Some(&self.compute_entrypoint),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            }),
        )
    }
}

pub type GpuComputePipelinePoolAccessor<'a> =
    StaticResourcePoolReadLockAccessor<'a, GpuComputePipelineHandle, wgpu::ComputePipeline>;

#[derive(Default)]
pub struct GpuComputePipelinePool {
    pool: StaticResourcePool<GpuComputePipelineHandle, ComputePipelineDesc, wgpu::ComputePipeline>,
}

impl GpuComputePipelinePool {
    pub fn get_or_create(
        &self,
        ctx: &RenderContext,
        desc: &ComputePipelineDesc,
    ) -> GpuComputePipelineHandle {
        self.pool.get_or_create(desc, |desc| {
            match desc.create_compute_pipeline(
                &ctx.device,
                &ctx.gpu_resources.pipeline_layouts,
                &ctx.gpu_resources.shader_modules,
            ) {
                Ok(compute_pipeline) => compute_pipeline,
                Err(err) => {
                    // TODO(ChristianSchott): handle this properly
                    panic!("Compute pipeline creation failed. {}", err);
                }
            }
        })
    }

    pub fn begin_frame(
        &mut self,
        device: &wgpu::Device,
        frame_index: u64,
        shader_modules: &GpuShaderModulePool,
        pipeline_layouts: &GpuPipelineLayoutPool,
    ) {
        re_tracing::profile_function!();
        self.pool.current_frame_index = frame_index;

        // Recompile render pipelines referencing shader modules that have been recompiled this frame.
        self.pool.recreate_resources(|desc| {
            let frame_created = {
                let shader_modules = shader_modules.resources();
                shader_modules
                    .get_statistics(desc.compute_handle)
                    .map(|sm| sm.frame_created)
                    .unwrap_or(0)
            };
            // The frame counter just got bumped by one. So any shader that has `frame_created`,
            // equal the current frame now, must have been recompiled since the user didn't have a
            // chance yet to add new shaders for this frame!
            // (note that this assumes that shader `begin_frame` happens before pipeline `begin_frame`)
            if frame_created < frame_index {
                return None;
            }

            match desc.create_compute_pipeline(device, pipeline_layouts, shader_modules) {
                Ok(sm) => {
                    // We don't know yet if this actually succeeded.
                    // But it's good to get feedback to the user that _something_ happened!
                    re_log::info!(label = desc.label.get(), "recompiled compute pipeline");
                    Some(sm)
                }
                Err(err) => {
                    re_log::error!("Failed to compile compute pipeline: {}", err);
                    None
                }
            }
        });
    }

    /// Locks the resource pool for resolving handles.
    ///
    /// While it is locked, no new resources can be added.
    pub fn resources(
        &self,
    ) -> StaticResourcePoolReadLockAccessor<'_, GpuComputePipelineHandle, wgpu::ComputePipeline>
    {
        self.pool.resources()
    }

    pub fn num_resources(&self) -> usize {
        self.pool.num_resources()
    }
}
