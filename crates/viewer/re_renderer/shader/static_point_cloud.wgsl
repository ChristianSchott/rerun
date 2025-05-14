#import <./types.wgsl>
#import <./global_bindings.wgsl>
#import <./utils/srgb.wgsl>
#import <./utils/sphere_quad.wgsl>

struct VertexIn {
    @builtin(vertex_index) vertex_idx: u32,
    @location(0) position: vec3f,
    @location(1) color: vec4f, // gamma-space 0-1, unmultiplied
    @location(2) picking_instance_id: vec2u,
    @location(3) outline_mask_ids: vec2u,
};

struct DrawDataUniformBuffer {
    world_from_obj: mat4x4f,
    picking_layer_object_id: vec2u,
    outline_mask_ids: vec2u,
    point_size: vec4f,
};

@group(1) @binding(0)
var<uniform> draw_data: DrawDataUniformBuffer;

struct VertexOut {
    @builtin(position)
    position: vec4f,

    @location(0) @interpolate(perspective)
    world_position: vec3f,

    @location(1) @interpolate(flat)
    radius: f32,

    @location(2) @interpolate(flat)
    point_center: vec3f,

    // TODO(andreas): Color & picking layer instance are only used in some passes.
    // Once we have shader variant support we should remove the unused ones
    // (it's unclear how good shader compilers are at removing unused outputs and associated texture fetches)
    // TODO(andreas): Is fetching color & picking layer in the fragment shader maybe more efficient?
    // Yes, that's more fetches but all of these would be cache hits whereas vertex data pass through can be expensive, (especially on tiler architectures!)

    @location(3) @interpolate(flat)
    color: vec4f, // linear RGBA with unmulitplied/separate alpha

    @location(4) @interpolate(flat)
    picking_instance_id: vec2u,

    @location(5) @interpolate(flat)
    outline_mask_ids: vec2u,

    // @location(0)
    // color: vec4f, // 0-1 linear space with unmultiplied/separate alpha

    // @location(1) @interpolate(flat)
    // outline_mask_ids: vec2u,

    // @location(2) @interpolate(flat)
    // picking_layer_id: vec4u,
};

// @vertex
// fn vs_main(in_vertex: VertexIn) -> VertexOut {
//     var out: VertexOut;
//     out.position = frame.projection_from_world * draw_data.world_from_obj * vec4f(in_vertex.position, 1.0);
//     out.color = linear_from_srgba(in_vertex.color);
//     out.outline_mask_ids = in_vertex.outline_mask_ids; // draw_data.outline_mask_ids;
//     out.picking_layer_id = vec4u(draw_data.picking_layer_object_id, in_vertex.picking_instance_id);

//     return out;
// }

struct PointData {
    pos: vec3f,
    unresolved_radius: f32,
    color: vec4f,
    picking_instance_id: vec2u,
}

// Read and unpack data at a given location
fn read_data(in_vertex: VertexIn) -> PointData {
    var data: PointData;
    let pos_4d = draw_data.world_from_obj * vec4f(in_vertex.position.xyz, 1.0);
    data.pos = pos_4d.xyz / pos_4d.w;
    data.unresolved_radius = draw_data.point_size.x;
    data.color = in_vertex.color;
    data.picking_instance_id = in_vertex.picking_instance_id;
    return data;
}

@vertex
fn vs_main(in_vertex: VertexIn) -> VertexOut {
    // Read point data (valid for the entire quad)
    let point_data = read_data(in_vertex);

    // Span quad
    let camera_distance = distance(frame.camera_position, point_data.pos);
    let world_scale_factor = average_scale_from_transform(draw_data.world_from_obj); // TODO(andreas): somewhat costly, should precompute this
    let world_radius = unresolved_size_to_world(point_data.unresolved_radius, camera_distance, world_scale_factor); // +world_size_from_point_size(draw_data.radius_boost_in_ui_points, camera_distance);
    let quad = sphere_or_circle_quad_span(in_vertex.vertex_idx, point_data.pos, world_radius, true);

    // Output, transform to projection space and done.
    var out: VertexOut;
    out.position = frame.projection_from_world * vec4f(quad.pos_in_world, 1.0);
    out.color = point_data.color;
    out.radius = quad.point_resolved_radius;
    out.world_position = quad.pos_in_world;
    out.point_center = point_data.pos;
    out.picking_instance_id = point_data.picking_instance_id;
    out.outline_mask_ids = in_vertex.outline_mask_ids;

    return out;
}

fn circle_quad_coverage(world_position: vec3f, radius: f32, circle_center: vec3f) -> f32 {
    let circle_distance = distance(circle_center, world_position);
    let feathering_radius = fwidth(circle_distance) * 0.5;
    return smoothstep(radius + feathering_radius, radius - feathering_radius, circle_distance);
}

@fragment
fn fs_main_shaded(in: VertexOut) -> @location(0) vec4f {
    let cov = circle_quad_coverage(in.world_position, in.radius, in.point_center);
    if cov < 0.001 || in.color.a == 0 {
        discard;
    }
    return vec4f(in.color.rgb, cov);

    // if in.color.a == 0 {
    //     discard;
    // }
    // return in.color.rgba;
}

@fragment
fn fs_main_picking_layer(in: VertexOut) -> @location(0) vec4u {
    let cov = circle_quad_coverage(in.world_position, in.radius, in.point_center);
    if cov <= 0.5 || in.color.a == 0 {
        discard;
    }
    return vec4u(draw_data.picking_layer_object_id, in.picking_instance_id);

    // if in.color.a == 0 {
    //     discard;
    // }
    // return in.picking_layer_id;
}

@fragment
fn fs_main_outline_mask(in: VertexOut) -> @location(0) vec2u {
    // Output is an integer target so we can't use coverage even though
    // the target is anti-aliased.
    let cov = circle_quad_coverage(in.world_position, in.radius, in.point_center);
    if cov <= 0.5 || in.color.a == 0 {
        discard;
    }
    return in.outline_mask_ids;

    // if in.color.a == 0 {
    //     discard;
    // }
    // return in.outline_mask_ids;
}
