//! Demonstrates outline rendering.

use std::{f32, sync::Arc};

use glam::{Affine3A, UVec2, Vec3};
use re_renderer::{
    renderer::PersistentPointCloudDrawData,
    view_builder::{Projection, TargetConfiguration, ViewBuilder},
    CPUPointCloud, GPUPersistentPointCloud, OutlineConfig, OutlineMaskPreference,
    PickingLayerInstanceId, PickingLayerObjectId, Rgba32Unmul,
};
use winit::event::ElementState;

mod framework;

struct PointCloud {
    is_paused: bool,
    seconds_since_startup: f32,
    point_cloud: Arc<GPUPersistentPointCloud>,
}

struct FibonacciSphere {
    num_points: usize,
    index: usize,
    scale: f32,
}

impl FibonacciSphere {
    fn new(num_points: usize, scale: f32) -> Self {
        Self {
            num_points,
            index: 0,
            scale,
        }
    }
}

impl Iterator for FibonacciSphere {
    type Item = Vec3;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.num_points {
            return None;
        }

        let golden_ratio = (1.0 + 5.0_f32.sqrt()) / 2.0;
        let phi = 2.0 * f32::consts::PI * self.index as f32 / golden_ratio;
        let cos_theta = 1.0 - (2.0 * self.index as f32 + 1.0) / self.num_points as f32;
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

        self.index += 1;
        Some(
            Vec3::new(sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta)
            // + some fun
                * (((self.index as f32 / self.num_points as f32) * self.scale).sin() * 0.5
                    + 0.8f32)
                * 2f32,
        )
    }
}

impl ExactSizeIterator for FibonacciSphere {
    fn len(&self) -> usize {
        self.num_points - self.index
    }
}

impl framework::Example for PointCloud {
    fn title() -> &'static str {
        "Outlines"
    }

    fn new(re_ctx: &re_renderer::RenderContext) -> Self {
        let num_points = 50_000;
        let points = Box::from_iter(FibonacciSphere::new(num_points, 0.0));
        let colors = Box::from_iter((0..num_points).map(|i| {
            let v = i as f32 / num_points as f32;
            Rgba32Unmul([
                (v * 255.) as u8,
                ((1.0 - v) * 255.) as u8,
                (0.5 * (1.0 - v) * 255.) as u8,
                255,
            ])
        }));
        let picking = Box::from_iter((0..num_points).map(|i| PickingLayerInstanceId(i as u64)));
        let outline = Box::from_iter((0..num_points).map(|i| {
            UVec2::new(
                if i % 3 == 0 { 1 } else { 0 },
                if i % 7 == 0 { 1 } else { 0 },
            )
        }));

        let pc = CPUPointCloud {
            label: "point_cloud".into(),
            positions: &points,
            colors: &colors,
            picking: &picking,
            outline: &outline,
        };

        Self {
            is_paused: false,
            seconds_since_startup: 0.0,
            point_cloud: Arc::new(
                GPUPersistentPointCloud::new(re_ctx, pc).expect("Failed creating point cloud."),
            ),
        }
    }

    fn draw(
        &mut self,
        re_ctx: &re_renderer::RenderContext,
        resolution: [u32; 2],
        time: &framework::Time,
        pixels_per_point: f32,
    ) -> anyhow::Result<Vec<framework::ViewDrawResult>> {
        if !self.is_paused {
            self.seconds_since_startup += time.last_frame_duration.as_secs_f32();
        }
        let seconds_since_startup = self.seconds_since_startup;

        // activate rave
        {
            let num_points = self.point_cloud.point_count;
            let a = (seconds_since_startup * 2f32).floor() as u64 % 3 + 3;
            let b = (seconds_since_startup * 3f32).floor() as u64 % 7 + 2;
            let outline = Box::from_iter((0..num_points).map(|i| {
                UVec2::new(
                    if i % a == 0 { 1 } else { 0 },
                    if i % b == 0 { 1 } else { 0 },
                )
            }));
            self.point_cloud.update_outline(re_ctx, &outline)?;

            let colors = Box::from_iter((0..num_points).map(|i| {
                let v = ((i as f32 / num_points as f32) * 10.0 + seconds_since_startup * 6.0).sin()
                    * 0.5
                    + 0.5;
                Rgba32Unmul([
                    (v * 255.) as u8,
                    ((1.0 - v) * 255.) as u8,
                    (0.5 * (1.0 - v) * 255.) as u8,
                    255,
                ])
            }));
            self.point_cloud.update_color(re_ctx, &colors)?;

            let points = Box::from_iter(FibonacciSphere::new(
                num_points as usize,
                (seconds_since_startup * 0.1f32).sin() * 40f32,
            ));
            self.point_cloud.update_positions(re_ctx, &points)?;
        }

        // TODO(#1426): unify camera logic between examples.
        let camera_position = glam::vec3(2.5, 1.5, 0.5 + (seconds_since_startup * 0.5).sin() * 2.5);

        let mut view_builder = ViewBuilder::new(
            re_ctx,
            TargetConfiguration {
                name: "OutlinesDemo".into(),
                resolution_in_pixel: resolution,
                view_from_world: re_math::IsoTransform::look_at_rh(
                    camera_position,
                    glam::Vec3::ZERO,
                    glam::Vec3::Y,
                )
                .ok_or(anyhow::format_err!("invalid camera"))?,
                projection_from_view: Projection::Perspective {
                    vertical_fov: 70.0 * std::f32::consts::TAU / 360.0,
                    near_plane_distance: 0.01,
                    aspect_ratio: resolution[0] as f32 / resolution[1] as f32,
                },
                pixels_per_point,
                outline_config: //None,
                Some(OutlineConfig {
                    outline_radius_pixel: 2.0,
                    color_layer_a: re_renderer::Rgba::from_rgb(1.0, 0.6, 0.0),
                    color_layer_b: re_renderer::Rgba::from_rgba_unmultiplied(0.25, 0.3, 1.0, 0.8),
                }),
                ..Default::default()
            },
        );

        view_builder.queue_draw(PersistentPointCloudDrawData::new(
            re_ctx,
            self.point_cloud.clone(),
            Affine3A::IDENTITY,
            PickingLayerObjectId(0),
            true,
            0.005f32 + (seconds_since_startup.sin() * 0.005f32),
        )?);

        view_builder.queue_draw(re_renderer::renderer::GenericSkyboxDrawData::new(
            re_ctx,
            Default::default(),
        ));

        let command_buffer = view_builder.draw(re_ctx, re_renderer::Rgba::TRANSPARENT)?;

        Ok(vec![framework::ViewDrawResult {
            view_builder,
            command_buffer,
            target_location: glam::Vec2::ZERO,
        }])
    }

    fn on_key_event(&mut self, input: winit::event::KeyEvent) {
        if input.state == ElementState::Pressed
            && input.logical_key == winit::keyboard::Key::Named(winit::keyboard::NamedKey::Space)
        {
            self.is_paused ^= true;
        }
    }
}

fn main() {
    framework::start::<PointCloud>();
}
