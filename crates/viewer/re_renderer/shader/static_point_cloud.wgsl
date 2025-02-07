#import <./types.wgsl>
#import <./global_bindings.wgsl>
#import <./utils/srgb.wgsl>
#import <./utils/sphere_quad.wgsl>

struct VertexInColor {
    @builtin(vertex_index) vertex_idx: u32,
    @location(0) position: vec3f,
    @location(1) color: vec4f, // gamma-space 0-1, unmultiplied
};

struct VertexInData {
    @builtin(vertex_index) vertex_idx: u32,
    @location(0) position: vec3f,
    @location(1) color: vec4f, // gamma-space 0-1, unmultiplied
    @location(2) data: vec2u,
};

struct DrawDataUniformBuffer {
    world_from_obj: mat4x4f,
    picking_layer_object_id: vec2u,
    outline_mask_ids: vec2u,
    point_size: vec4f,
};

@group(1) @binding(0)
var<uniform> draw_data: DrawDataUniformBuffer;

struct VertexOutColor {
    @builtin(position)
    position: vec4f,

    @location(0) @interpolate(flat)
    color: vec4f, // linear RGBA with unmulitplied/separate alpha
};

struct VertexOutData {
    @builtin(position)
    position: vec4f,

    @location(0) @interpolate(flat)
    data: vec2u,
};

const TRIANGLE_VERTS = array<vec2f, 3>(
    vec2f( 1.0,  0.0),
    vec2f(-0.5,  0.86602540378),
    vec2f(-0.5, -0.86602540378),
);

// Read and unpack data at a given location
fn vertex_world_pos(vertex_idx: u32, position: vec3f) -> vec3f {

    let pos_4d = draw_data.world_from_obj * vec4f(position.xyz, 1.0);
    let pos = pos_4d.xyz / pos_4d.w;

    // Span quad
    // let camera_distance = distance(frame.camera_position, data.pos);
    // let world_scale_factor = average_scale_from_transform(draw_data.world_from_obj); // TODO(andreas): somewhat costly, should precompute this
    // let world_radius = unresolved_size_to_world(draw_data.point_size.x, camera_distance, world_scale_factor); // +world_size_from_point_size(draw_data.radius_boost_in_ui_points, camera_distance);
    // let quad = sphere_or_circle_quad_span(vertex_idx, data.pos, draw_data.point_size.x, true);
    // data.pos_in_world = quad.pos_in_world;
    // data.point_resolved_radius = quad.point_resolved_radius;

    let vert_pos = TRIANGLE_VERTS[vertex_idx % 3u];

    return circle_quad(pos, draw_data.point_size.x, vert_pos.y, vert_pos.x);
}

@vertex
fn vs_main_color(in_vertex: VertexInColor) -> VertexOutColor {
    let pos_in_world = vertex_world_pos(in_vertex.vertex_idx, in_vertex.position);

    var out: VertexOutColor;
    out.position = frame.projection_from_world * vec4f(pos_in_world, 1.0);
    if in_vertex.color.a == 0 {
        out.position = vec4f(-10.0, -10.0, -10.0, 0.0);
    }
    out.color = in_vertex.color;

    return out;
}

@vertex
fn vs_main_data(in_vertex: VertexInData) -> VertexOutData {
    let pos_in_world = vertex_world_pos(in_vertex.vertex_idx, in_vertex.position);

    var out: VertexOutData;
    out.position = frame.projection_from_world * vec4f(pos_in_world, 1.0);
    if in_vertex.color.a == 0 {
        out.position = vec4f(-10.0, -10.0, -10.0, 0.0);
    }
    out.data = in_vertex.data;

    return out;
}

fn circle_quad_coverage(world_position: vec3f, radius: f32, circle_center: vec3f) -> f32 {
    let circle_distance = distance(circle_center, world_position);
    let feathering_radius = fwidth(circle_distance) * 0.5;
    return smoothstep(radius + feathering_radius, radius - feathering_radius, circle_distance);
}

@fragment
fn fs_main_shaded(in: VertexOutColor) -> @location(0) vec4f {
    let cov = 1.0;
    // let cov = circle_quad_coverage(in.world_position, in.radius, in.point_center);
    // if cov < 0.001 {
    //     discard;
    // }
    return vec4f(in.color.rgb, cov);

    // if in.color.a == 0 {
    //     discard;
    // }
    // return in.color.rgba;
}

@fragment
fn fs_main_picking_layer(in: VertexOutData) -> @location(0) vec4u {
    // let cov = circle_quad_coverage(in.world_position, in.radius, in.point_center);
    // if cov <= 0.5 {
    //     discard;
    // }
    return vec4u(draw_data.picking_layer_object_id, in.data);

    // if in.color.a == 0 {
    //     discard;
    // }
    // return in.picking_layer_id;
}

@fragment
fn fs_main_outline_mask(in: VertexOutData) -> @location(0) vec2u {
    // Output is an integer target so we can't use coverage even though
    // the target is anti-aliased.
    // let cov = circle_quad_coverage(in.world_position, in.radius, in.point_center);
    // if cov <= 0.5 {
    //     discard;
    // }
    return in.data;

    // if in.color.a == 0 {
    //     discard;
    // }
    // return in.outline_mask_ids;
}
