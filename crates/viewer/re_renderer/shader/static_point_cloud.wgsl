#import <./types.wgsl>
#import <./global_bindings.wgsl>
#import <./utils/srgb.wgsl>

struct VertexIn {
    @location(0) position: vec3f,
    @location(1) color: vec4f, // gamma-space 0-1, unmultiplied
    @location(2) picking_instance_id: vec2u,
    @location(3) outline_mask_ids: vec2u,
};

struct DrawDataUniformBuffer {
    world_from_obj: mat4x4f,
    outline_mask_ids: vec2u,
    picking_layer_object_id: vec2u,
};

@group(1) @binding(0)
var<uniform> draw_data: DrawDataUniformBuffer;

struct VertexOut {
    @builtin(position)
    position: vec4f,

    @location(0)
    color: vec4f, // 0-1 linear space with unmultiplied/separate alpha

    @location(1) @interpolate(flat)
    outline_mask_ids: vec2u,

    @location(2) @interpolate(flat)
    picking_layer_id: vec4u,
};

@vertex
fn vs_main(in_vertex: VertexIn) -> VertexOut {
    var out: VertexOut;
    out.position = frame.projection_from_world * draw_data.world_from_obj * vec4f(in_vertex.position, 1.0);
    out.color = linear_from_srgba(in_vertex.color);
    out.outline_mask_ids = in_vertex.outline_mask_ids; // draw_data.outline_mask_ids;
    out.picking_layer_id = vec4u(draw_data.picking_layer_object_id, in_vertex.picking_instance_id);

    return out;
}

@fragment
fn fs_main_shaded(in: VertexOut) -> @location(0) vec4f {
    return in.color.rgba;
}

@fragment
fn fs_main_picking_layer(in: VertexOut) -> @location(0) vec4u {
    if in.color.a < 0.001 {
        discard;
    }
    return in.picking_layer_id;
}

    @fragment
fn fs_main_outline_mask(in: VertexOut) -> @location(0) vec2u {
    if in.color.a < 0.001 {
        discard;
    }
    return in.outline_mask_ids;
}
