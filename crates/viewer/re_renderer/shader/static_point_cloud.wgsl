#import <./types.wgsl>
#import <./global_bindings.wgsl>
#import <./utils/srgb.wgsl>

struct VertexIn {
    @location(0) position: vec3f,
    @location(1) color: vec4f, // gamma-space 0-1, unmultiplied
    // @location(2) normal: vec3f,
    // @location(3) texcoord: vec2f,
};

struct VertexOut {
    @builtin(position)
    position: vec4f,

    @location(0)
    color: vec4f, // 0-1 linear space with unmultiplied/separate alpha

    // @location(1) @interpolate(flat)
    // outline_mask_ids: vec2u,

    // @location(2) @interpolate(flat)
    // picking_layer_id: vec4u,
};

@vertex
fn vs_main(in_vertex: VertexIn) -> VertexOut {
    var out: VertexOut;
    out.position = frame.projection_from_world * vec4f(in_vertex.position, 1.0);
    out.color = linear_from_srgba(in_vertex.color);
    // out.outline_mask_ids = in_instance.outline_mask_ids;
    // out.picking_layer_id = in_instance.picking_layer_id;

    return out;
}

@fragment
fn fs_main_shaded(in: VertexOut) -> @location(0) vec4f {
    return vec4f(in.color.rgb, 1.0);
}

// @fragment
// fn fs_main_picking_layer(in: VertexOut) -> @location(0) vec4u {
//     return in.picking_layer_id;
// }

// @fragment
// fn fs_main_outline_mask(in: VertexOut) -> @location(0) vec2u {
//     return in.outline_mask_ids;
// }
