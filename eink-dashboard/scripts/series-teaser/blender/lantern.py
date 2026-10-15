"""The Long Drift teaser v2: the 3D shots, built and rendered in Blender (DECISIONS #77,
docs/series/TEASER.md). Everything is made from primitives here: the Last Lantern (our own
design), the dusk outpost, the starfield. No downloaded or generated models, no .blend file.

  Headless render of one shot into TEASER_DIR/shots/<shot>/f0001.png ...:
    /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup \
        -P lantern.py -- render cold|arrival|jump|wide [first last]
  A still (frame N of a shot) to TEASER_DIR/shots/still-<shot>.png:
    ... -P lantern.py -- still arrival 160
  Inside a running Blender (the MCP), exec() this file and call build("arrival").

Look: toon-banded shading (Shader to RGB), EEVEE volumetrics, a Kuwahara painterly pass and a
glow in the compositor. 1920x804 (the 2.39:1 picture area; teaser.py adds the bars), 24 fps.
"""
import math
import os
import random
import sys

import bpy
from mathutils import Euler, Matrix, Vector

FPS = 24
RX, RY = 1920, 804
TEASER_DIR = os.path.expanduser(os.environ.get("TEASER_DIR", "~/series-work/teaser-v2"))
# Shot name -> number of frames. Frame 1 is the shot's first frame in the teaser timeline.
SHOTS = {"cold": 252, "arrival": 226, "jump": 30, "wide": 180}

# ---------------------------------------------------------------- scene plumbing


def reset():
    for coll in (bpy.data.objects, bpy.data.meshes, bpy.data.materials, bpy.data.lights,
                 bpy.data.cameras, bpy.data.curves, bpy.data.node_groups, bpy.data.images):
        for block in list(coll):
            coll.remove(block)
    for c in list(bpy.data.collections):
        bpy.data.collections.remove(c)
    sc = bpy.context.scene
    if sc.world is None:
        sc.world = bpy.data.worlds.new("World")
    return sc


def link(obj):
    bpy.context.scene.collection.objects.link(obj)
    return obj


def empty(name, loc=(0, 0, 0), parent=None):
    e = link(bpy.data.objects.new(name, None))
    e.location = loc
    e.parent = parent
    return e


def prim(kind, name, loc=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1), mat=None, parent=None,
         bevel=0.0, smooth=True, **kw):
    """A primitive with its rotation and (world-axis) scale baked into the mesh."""
    op = {"cube": bpy.ops.mesh.primitive_cube_add, "cyl": bpy.ops.mesh.primitive_cylinder_add,
          "cone": bpy.ops.mesh.primitive_cone_add, "sphere": bpy.ops.mesh.primitive_uv_sphere_add,
          "ico": bpy.ops.mesh.primitive_ico_sphere_add, "torus": bpy.ops.mesh.primitive_torus_add,
          "plane": bpy.ops.mesh.primitive_plane_add}[kind]
    op(**kw)
    o = bpy.context.active_object
    o.name = name
    o.data.transform(Matrix.Diagonal((*scale, 1)) @ Euler(rot).to_matrix().to_4x4())
    o.location = loc
    if parent is not None:
        o.parent = parent
    if mat is not None:
        o.data.materials.append(mat)
    if smooth:
        for p in o.data.polygons:
            p.use_smooth = True
    if bevel:
        m = o.modifiers.new("bevel", "BEVEL")
        m.width = bevel
        m.segments = 2
        m.limit_method = "ANGLE"
        o.modifiers.new("wn", "WEIGHTED_NORMAL").keep_sharp = True
    return o


def strut(name, a, b, r, mat, parent):
    """A cylinder from point a to point b."""
    a, b = Vector(a), Vector(b)
    d = b - a
    o = prim("cyl", name, loc=(a + b) / 2, mat=mat, parent=parent, radius=r, depth=d.length, vertices=12)
    o.rotation_mode = "QUATERNION"
    o.rotation_quaternion = d.to_track_quat("Z", "Y")
    return o


def key(obj, path, frame, value, index=-1):
    if index >= 0:
        getattr(obj, path)[index] = value
    else:
        setattr(obj, path, value)
    obj.keyframe_insert(path, index=index, frame=frame)


def smooth_keys(*objs, interp="BEZIER"):
    for o in objs:
        ad = o.animation_data
        if ad and ad.action:
            for fc in getattr(ad.action, "fcurves", []):
                for k in fc.keyframe_points:
                    k.interpolation = interp


def nodes_of(mat):
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    return nt, nt.nodes, nt.links


# ---------------------------------------------------------------- materials


def toon(name, color, grime=0.35, panels=False, emit=None, emit_strength=0.0, ambient=0.06, soft=False,
         grime_color=None):
    """Banded toon shading that keeps some of the light's colour, over a weathered base."""
    mat = bpy.data.materials.new(name)
    nt, N, L = nodes_of(mat)
    out = N.new("ShaderNodeOutputMaterial")
    coord = N.new("ShaderNodeTexCoord")
    # Weathered base: blotches, vertical streaks, optional panel seams.
    noise = N.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 3.5
    noise.inputs["Detail"].default_value = 8
    L.new(coord.outputs["Object"], noise.inputs["Vector"])
    streak_map = N.new("ShaderNodeMapping")
    streak_map.inputs["Scale"].default_value = (14, 14, 1.2)
    L.new(coord.outputs["Object"], streak_map.inputs["Vector"])
    streak = N.new("ShaderNodeTexNoise")
    streak.inputs["Scale"].default_value = 2.0
    streak.inputs["Detail"].default_value = 4
    L.new(streak_map.outputs["Vector"], streak.inputs["Vector"])
    dirt = N.new("ShaderNodeMath")
    dirt.operation = "MULTIPLY"
    L.new(noise.outputs["Fac"], dirt.inputs[0])
    L.new(streak.outputs["Fac"], dirt.inputs[1])
    dramp = N.new("ShaderNodeValToRGB")
    dramp.color_ramp.elements[0].position = 0.18
    dramp.color_ramp.elements[1].position = 0.42
    L.new(dirt.outputs[0], dramp.inputs["Fac"])
    # Patches: big, hard-edged blotches (rust, primer, replaced plates).
    patch = N.new("ShaderNodeTexNoise")
    patch.inputs["Scale"].default_value = 1.1
    patch.inputs["Detail"].default_value = 5
    patch.inputs["Roughness"].default_value = 0.65
    L.new(coord.outputs["Object"], patch.inputs["Vector"])
    pramp = N.new("ShaderNodeValToRGB")
    pramp.color_ramp.elements[0].position = 0.56
    pramp.color_ramp.elements[1].position = 0.6
    L.new(patch.outputs["Fac"], pramp.inputs["Fac"])
    both = N.new("ShaderNodeMath")
    both.operation = "MAXIMUM"
    L.new(dramp.outputs["Color"], both.inputs[0])
    L.new(pramp.outputs["Color"], both.inputs[1])
    base = N.new("ShaderNodeMix")
    base.data_type = "RGBA"
    base.inputs["A"].default_value = (*color, 1)
    base.inputs["B"].default_value = (*grime_color, 1) if grime_color else (color[0] * 0.38, color[1] * 0.34, color[2] * 0.3, 1)
    gfac = N.new("ShaderNodeMath")
    gfac.operation = "MULTIPLY"
    gfac.inputs[1].default_value = grime
    L.new(both.outputs[0], gfac.inputs[0])
    L.new(gfac.outputs[0], base.inputs["Factor"])
    col = base.outputs["Result"]
    if panels:
        wave = N.new("ShaderNodeTexWave")
        wave.bands_direction = "X"
        wave.inputs["Scale"].default_value = 0.16
        wave.inputs["Distortion"].default_value = 0
        L.new(coord.outputs["Object"], wave.inputs["Vector"])
        seam = N.new("ShaderNodeValToRGB")
        seam.color_ramp.interpolation = "CONSTANT"
        seam.color_ramp.elements[0].color = (0.45, 0.45, 0.45, 1)
        seam.color_ramp.elements[1].position = 0.04
        L.new(wave.outputs["Fac"], seam.inputs["Fac"])
        mul = N.new("ShaderNodeMix")
        mul.data_type = "RGBA"
        mul.blend_type = "MULTIPLY"
        mul.inputs["Factor"].default_value = 1
        L.new(col, mul.inputs["A"])
        L.new(seam.outputs["Color"], mul.inputs["B"])
        col = mul.outputs["Result"]
    # Lighting -> bands.
    diff = N.new("ShaderNodeBsdfDiffuse")
    s2r = N.new("ShaderNodeShaderToRGB")
    L.new(diff.outputs[0], s2r.inputs[0])
    band = N.new("ShaderNodeValToRGB")
    cr = band.color_ramp
    cr.interpolation = "EASE" if soft else "CONSTANT"
    cr.elements[0].color = (ambient, ambient, ambient, 1)
    cr.elements[1].position = 0.05
    cr.elements[1].color = (0.42, 0.42, 0.42, 1)
    e = cr.elements.new(0.28)
    e.color = (0.8, 0.8, 0.8, 1)
    e = cr.elements.new(0.75)
    e.color = (1.25, 1.25, 1.25, 1)
    L.new(s2r.outputs["Color"], band.inputs["Fac"])
    tint = N.new("ShaderNodeMix")
    tint.data_type = "RGBA"
    tint.inputs["Factor"].default_value = 0.4
    L.new(band.outputs["Color"], tint.inputs["A"])
    L.new(s2r.outputs["Color"], tint.inputs["B"])
    lit = N.new("ShaderNodeMix")
    lit.data_type = "RGBA"
    lit.blend_type = "MULTIPLY"
    lit.inputs["Factor"].default_value = 1
    L.new(col, lit.inputs["A"])
    L.new(tint.outputs["Result"], lit.inputs["B"])
    em = N.new("ShaderNodeEmission")
    L.new(lit.outputs["Result"], em.inputs["Color"])
    shader = em.outputs[0]
    if emit is not None:
        em2 = N.new("ShaderNodeEmission")
        em2.inputs["Color"].default_value = (*emit, 1)
        em2.inputs["Strength"].default_value = emit_strength
        add = N.new("ShaderNodeAddShader")
        L.new(shader, add.inputs[0])
        L.new(em2.outputs[0], add.inputs[1])
        shader = add.outputs[0]
    L.new(shader, out.inputs["Surface"])
    return mat


def glow(name, color, strength):
    mat = bpy.data.materials.new(name)
    nt, N, L = nodes_of(mat)
    out = N.new("ShaderNodeOutputMaterial")
    em = N.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*color, 1)
    em.inputs["Strength"].default_value = strength
    L.new(em.outputs[0], out.inputs["Surface"])
    return mat


def halo(name, color, strength, power=3.0):
    """A soft glow ball: emission fading to transparent at the silhouette."""
    mat = bpy.data.materials.new(name)
    nt, N, L = nodes_of(mat)
    out = N.new("ShaderNodeOutputMaterial")
    lw = N.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.5
    inv = N.new("ShaderNodeMath")
    inv.operation = "SUBTRACT"
    inv.inputs[0].default_value = 1
    L.new(lw.outputs["Facing"], inv.inputs[1])
    pw = N.new("ShaderNodeMath")
    pw.operation = "POWER"
    pw.inputs[1].default_value = power
    L.new(inv.outputs[0], pw.inputs[0])
    em = N.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*color, 1)
    em.inputs["Strength"].default_value = strength
    tr = N.new("ShaderNodeBsdfTransparent")
    mix = N.new("ShaderNodeMixShader")
    L.new(pw.outputs[0], mix.inputs["Fac"])
    L.new(tr.outputs[0], mix.inputs[1])
    L.new(em.outputs[0], mix.inputs[2])
    L.new(mix.outputs[0], out.inputs["Surface"])
    if hasattr(mat, "surface_render_method"):
        mat.surface_render_method = "BLENDED"
    return mat


# ---------------------------------------------------------------- the Last Lantern
# Shape from the owner's concept sheet (2026-10-14): a faceted octagonal hull that tapers to a
# blunt prow, a raised cab on a short neck, a dorsal spine with a dish, a heavy armoured engine
# cowl with a finned drum and two swept spikes, three mismatched pods (white, rust, teal) on a
# belly rack, four chunky legs that fold away in flight.


def octagon(w, h, c=0.28):
    """A chamfered rectangle in the YZ plane, as 8 (y, z) points."""
    k = c * min(w, h)
    y, z = w / 2, h / 2
    return [(y, z - k), (y - k, z), (-y + k, z), (-y, z - k), (-y, -z + k), (-y + k, -z), (y - k, -z), (y, -z + k)]


def loft(name, sections, mat, parent=None, chamfer=0.28, bevel=0.06):
    """A faceted hull through octagonal sections [(x, w, h, zc), ...], capped at both ends."""
    verts, faces = [], []
    for x, w, h, zc in sections:
        verts += [(x, y, z + zc) for y, z in octagon(w, h, chamfer)]
    n = len(sections)
    for i in range(n - 1):
        for j in range(8):
            a, b = i * 8 + j, i * 8 + (j + 1) % 8
            faces.append((a, b, b + 8, a + 8))
    faces.append(tuple(range(7, -1, -1)))
    faces.append(tuple(range((n - 1) * 8, n * 8)))
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.validate()
    me.update()
    o = link(bpy.data.objects.new(name, me))
    o.parent = parent
    o.data.materials.append(mat)
    if bevel:
        m = o.modifiers.new("bevel", "BEVEL")
        m.width = bevel
        m.segments = 2
        o.modifiers.new("wn", "WEIGHTED_NORMAL").keep_sharp = True
    return o


def build_ship(engine_strength=1.0):
    """~29 m, nose toward +X. Returns (root, parts dict)."""
    rnd = random.Random(26)
    root = empty("Lantern")
    hull = toon("hull", (0.8, 0.76, 0.67), grime=0.85, panels=True, grime_color=(0.6, 0.3, 0.13))
    dark = toon("hull_dark", (0.27, 0.25, 0.23), grime=0.5, grime_color=(0.42, 0.22, 0.12))
    metal = toon("metal", (0.4, 0.39, 0.38), grime=0.4)
    pods = [toon("pod_white", (0.82, 0.78, 0.7), grime=0.6, grime_color=(0.5, 0.42, 0.32)),
            toon("pod_rust", (0.66, 0.32, 0.15), grime=0.5),
            toon("pod_teal", (0.3, 0.48, 0.47), grime=0.55)]
    glass = toon("glass", (0.05, 0.07, 0.09), grime=0.0, emit=(0.55, 0.75, 1.0), emit_strength=0.5)
    win = glow("window", (1.0, 0.72, 0.4), 5.0)
    blue = glow("engine", (0.35, 0.65, 1.0), 40.0)
    # Everything but the legs and ramp hangs off `body`, stretched to the sheet's chunky proportions.
    body = empty("body", parent=root)
    body.scale = (1, 1.12, 1.3)
    # Main hull: blunt prow, widening aft.
    loft("hull_main", [(14.2, 1.7, 1.0, -0.25), (13.3, 2.9, 1.8, -0.12), (11.0, 4.0, 2.55, 0.0),
                       (6.0, 4.6, 2.95, 0.1), (0.0, 4.85, 3.3, 0.22), (-6.6, 5.05, 3.6, 0.32)], hull, body)
    # Chined keel under the forward hull.
    loft("keel", [(12.5, 1.2, 0.5, -1.25), (8.0, 2.6, 0.7, -1.45), (2.0, 3.0, 0.7, -1.55)], dark, body, bevel=0.04)
    # Engine cowl: an armoured block bigger than the hull, banded in dark metal.
    loft("cowl", [(-6.4, 5.6, 3.9, 0.25), (-7.2, 6.3, 4.6, 0.3), (-10.6, 6.3, 4.6, 0.3), (-11.3, 5.6, 4.0, 0.3)], hull, body)
    for i, x in enumerate((-7.25, -10.55)):
        loft(f"cowl_band{i}", [(x + 0.25, 6.45, 4.75, 0.3), (x - 0.25, 6.45, 4.75, 0.3)], dark, body, bevel=0.03)
    # Finned engine drum and its glow.
    prim("cyl", "drum", loc=(-12.4, 0, 0.3), rot=(0, math.pi / 2, 0), mat=dark, parent=body,
         radius=2.0, depth=2.6, vertices=32)
    for i in range(28):
        a = i / 28 * 2 * math.pi
        prim("cube", f"fin{i}", loc=(-12.5, 2.18 * math.cos(a), 0.3 + 2.18 * math.sin(a)), rot=(a, 0, 0),
             scale=(1.05, 0.06, 0.25), mat=metal, parent=body, size=2, smooth=False)
    prim("torus", "nozzle_lip", loc=(-13.7, 0, 0.3), rot=(0, math.pi / 2, 0), mat=metal, parent=body,
         major_radius=1.85, minor_radius=0.22)
    prim("cyl", "engine_core", loc=(-13.72, 0, 0.3), rot=(0, math.pi / 2, 0), mat=blue, parent=body,
         radius=1.65, depth=0.08, vertices=40)
    gl = prim("sphere", "engine_halo", loc=(-14.3, 0, 0.3), scale=(1.15, 1, 1),
              mat=halo("engine_halo", (0.3, 0.6, 1.0), 9.0), parent=body, radius=2.1)
    gl.visible_shadow = False
    light = bpy.data.lights.new("engine_light", "POINT")
    light.color = (0.35, 0.62, 1.0)
    light.energy = 60000 * engine_strength
    light.shadow_soft_size = 1.6
    el = link(bpy.data.objects.new("engine_light", light))
    el.location = (-15.6, 0, 0.3)
    el.parent = body
    # Two swept spikes off the cowl, top and bottom.
    for nm, z, sz in (("spike_top", 2.35, 1), ("spike_low", -1.75, -1)):
        loft(nm, [(-8.6, 0.5, 0.9, z), (-14.6, 0.08, 0.12, z + sz * 0.9)], dark, body, chamfer=0.4, bevel=0)
    # Raised cab on a short neck, a band of windows round its front.
    loft("neck", [(10.0, 1.6, 1.3, 1.75), (7.4, 2.0, 1.3, 1.75)], dark, body, bevel=0.04)
    loft("cab", [(10.6, 1.9, 1.15, 2.45), (10.1, 2.5, 1.3, 2.5), (7.8, 2.6, 1.3, 2.5), (7.3, 2.2, 1.1, 2.45)], hull, body)
    loft("cab_windows", [(10.42, 2.05, 0.36, 2.62), (9.6, 2.58, 0.36, 2.62)], glass, body, chamfer=0.1, bevel=0)
    prim("cube", "cab_roof", loc=(8.9, 0, 3.25), scale=(1.0, 0.8, 0.12), mat=dark, parent=body, size=2, bevel=0.04)
    strut("cab_antenna", (8.4, 0.6, 3.3), (8.4, 0.6, 4.4), 0.04, metal, body)
    # Dorsal spine and hump, grille, dish on a mast.
    prim("cube", "spine", loc=(-0.6, 0, 1.85), scale=(5.6, 1.15, 0.38), mat=hull, parent=body, size=2, bevel=0.1)
    prim("cube", "hump", loc=(-3.6, 0, 2.35), scale=(2.4, 1.35, 0.32), mat=dark, parent=body, size=2, bevel=0.1)
    prim("cube", "grille", loc=(3.4, 0, 2.3), scale=(1.1, 0.75, 0.15), mat=dark, parent=body, size=2, bevel=0.03)
    for i in range(7):
        prim("cube", f"grille_slat{i}", loc=(2.6 + i * 0.27, 0, 2.47), scale=(0.05, 0.7, 0.04), mat=metal,
             parent=body, size=2, smooth=False)
    strut("mast", (0.8, 0, 2.2), (0.8, 0, 3.6), 0.12, metal, body)
    dish = empty("dish", loc=(0.8, 0, 3.75), parent=body)
    dish.rotation_euler = (0.25, -0.9, 0)
    prim("cone", "dish_bowl", loc=(0, 0, 0.18), mat=metal, parent=dish, radius1=0.15, radius2=0.95,
         depth=0.35, vertices=24)
    prim("cyl", "dish_feed", loc=(0, 0, 0.55), mat=dark, parent=dish, radius=0.05, depth=0.7, vertices=8)
    # Belly rack and the three mismatched pods.
    prim("cube", "rack", loc=(-0.8, 0, -1.6), scale=(6.0, 1.75, 0.22), mat=dark, parent=body, size=2, bevel=0.05)
    for i, (x, m) in enumerate(zip((3.3, -0.7, -4.7), pods)):
        prim("cube", f"pod{i}", loc=(x, 0, -2.55), scale=(1.85, 1.35, 0.72), mat=m, parent=body, bevel=0.14, size=2)
        for s_ in (1, -1):
            prim("cube", f"pod{i}_clamp{s_}", loc=(x + s_ * 1.2, 0, -1.95), scale=(0.15, 1.45, 0.2), mat=metal,
                 parent=body, size=2, bevel=0.03)
    # Flank clutter: pipes, boxes, vents, a side door.
    for side in (1, -1):
        y = side * 2.5
        strut(f"pipe{side}", (-6.3, y, -0.55), (8.0, y * 0.94, -0.55), 0.13, metal, body)
        strut(f"pipe{side}b", (-6.3, y * 0.98, -0.95), (4.0, y * 0.98, -0.95), 0.09, dark, body)
        strut(f"pipe{side}c", (-6.0, y * 0.97, 0.9), (1.5, y * 0.97, 0.9), 0.08, metal, body)
        for j in range(9):
            x = rnd.uniform(-6, 8)
            z = rnd.uniform(-0.3, 1.0)
            prim("cube", f"greeble{side}{j}", loc=(x, y * 0.99, z),
                 scale=(rnd.uniform(0.25, 0.8), 0.12, rnd.uniform(0.12, 0.35)), mat=rnd.choice([dark, metal, dark]),
                 parent=body, size=2, smooth=False)
        for j in range(4):
            x = rnd.uniform(-9.6, -8.2)
            prim("cube", f"cowl_greeble{side}{j}", loc=(x, side * 3.18, rnd.uniform(-1.0, 1.6)),
                 scale=(0.3, 0.1, rnd.uniform(0.2, 0.5)), mat=dark, parent=body, size=2, smooth=False)
        for j, x in enumerate([2.0, 2.8, 3.6, 4.4]):
            prim("cube", f"win{side}{j}", loc=(x, side * 2.41, 0.55), scale=(0.17, 0.05, 0.12), mat=win, parent=body,
                 size=2, smooth=False)
    prim("cube", "door", loc=(9.0, 2.12, -0.2), scale=(0.55, 0.05, 0.85), mat=dark, parent=body, size=2)
    # Landing legs: they fold away in flight (gear()).
    struts = empty("struts", loc=(0, 0, -1.45), parent=root)
    for x in (6.3, -6.4):
        for side in (1, -1):
            top = Vector((x, side * 1.9, 0))
            knee = Vector((x, side * 2.55, -1.9))
            foot = Vector((x, side * 2.6, -3.45))
            prim("cube", f"leg_mount{x}{side}", loc=top + Vector((0, 0, -0.15)), scale=(0.45, 0.35, 0.3),
                 mat=dark, parent=struts, size=2, bevel=0.05)
            strut(f"leg{x}{side}", top, knee, 0.33, metal, struts)
            strut(f"shin{x}{side}", knee, foot, 0.24, dark, struts)
            strut(f"piston{x}{side}", top + Vector((0.5, 0, -0.2)), knee + Vector((0.3, 0, 0.2)), 0.1, metal, struts)
            prim("cyl", f"foot{x}{side}", loc=foot + Vector((0, 0, -0.08)), mat=dark, parent=struts,
                 radius=0.75, depth=0.25, vertices=12, bevel=0.04)
    # Ramp under the prow: hinged at its rear edge, drops forward to the ground.
    ramp = empty("ramp", loc=(8.6, 0, -1.75), parent=root)
    prim("cube", "ramp_plate", loc=(2.3, 0, 0), scale=(2.3, 0.9, 0.07), mat=metal, parent=ramp, size=2)
    for i in range(6):
        prim("cube", f"ramp_step{i}", loc=(0.5 + i * 0.7, 0, 0.09), scale=(0.06, 0.85, 0.03), mat=dark, parent=ramp,
             size=2, smooth=False)
    hatch = prim("cube", "hatch_light", loc=(10.7, 0, -1.72), scale=(2.0, 0.85, 0.02),
                 mat=glow("hatch", (1.0, 0.72, 0.42), 0.0), parent=root, size=2)
    return root, {"struts": struts, "ramp": ramp, "hatch": hatch, "engine_light": el, "halo": gl,
                  "engine_mat": blue}


def gear(parts, shown, frame=None):
    """Show or hide every landing-gear part, optionally keyed at `frame`."""
    for o in parts["struts"].children:
        o.hide_render = not shown
        o.hide_viewport = not shown
        if frame is not None:
            o.keyframe_insert("hide_render", frame=frame)
            o.keyframe_insert("hide_viewport", frame=frame)


def crew(parent, origin):
    """Three silhouettes: pilot, droid, mechanic. Archetypes only."""
    dark = toon("crew", (0.08, 0.07, 0.07), grime=0.0, ambient=0.02)
    eye = glow("droid_eye", (1.0, 0.45, 0.15), 12)
    ox, oy, oz = origin
    # Pilot: tall, long coat.
    prim("cone", "pilot_coat", loc=(ox, oy + 1.0, oz + 0.62), mat=dark, parent=parent, radius1=0.42, radius2=0.2, depth=1.25)
    prim("sphere", "pilot_head", loc=(ox, oy + 1.0, oz + 1.62), mat=dark, parent=parent, radius=0.16)
    prim("cyl", "pilot_chest", loc=(ox, oy + 1.0, oz + 1.33), mat=dark, parent=parent, radius=0.2, depth=0.45)
    # Droid: small, round head, one eye.
    prim("cube", "droid_body", loc=(ox + 0.2, oy, oz + 0.55), scale=(0.22, 0.26, 0.32), mat=dark, parent=parent, size=2, bevel=0.04)
    prim("sphere", "droid_head", loc=(ox + 0.2, oy, oz + 1.05), mat=dark, parent=parent, radius=0.2)
    prim("sphere", "droid_eye", loc=(ox + 0.38, oy, oz + 1.08), mat=eye, parent=parent, radius=0.06)
    for s in (1, -1):
        prim("cyl", f"droid_leg{s}", loc=(ox + 0.2, oy + s * 0.13, oz + 0.12), mat=dark, parent=parent, radius=0.05, depth=0.3)
    # Mechanic: stocky, a tool bag.
    prim("cyl", "mech_body", loc=(ox - 0.1, oy - 1.0, oz + 0.75), scale=(1, 1.2, 1), mat=dark, parent=parent, radius=0.3, depth=1.5)
    prim("sphere", "mech_head", loc=(ox - 0.1, oy - 1.0, oz + 1.65), mat=dark, parent=parent, radius=0.17)
    prim("cube", "mech_bag", loc=(ox, oy - 1.42, oz + 0.45), scale=(0.25, 0.12, 0.18), mat=dark, parent=parent, size=2)


# ---------------------------------------------------------------- worlds


def world_space(nebula=0.6):
    w = bpy.context.scene.world
    nt, N, L = nodes_of(w)
    out = N.new("ShaderNodeOutputWorld")
    coord = N.new("ShaderNodeTexCoord")
    total = None
    # Stars: three layers of Voronoi points on the view direction.
    for scale, thresh, bright in ((420, 0.07, 2.0), (170, 0.06, 4.0), (60, 0.04, 9.0)):
        vor = N.new("ShaderNodeTexVoronoi")
        vor.inputs["Scale"].default_value = scale
        vor.inputs["Randomness"].default_value = 1.0
        L.new(coord.outputs["Generated"], vor.inputs["Vector"])
        ramp = N.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].position = 0.0
        ramp.color_ramp.elements[0].color = (bright, bright, bright * 1.05, 1)
        ramp.color_ramp.elements[1].position = thresh
        ramp.color_ramp.elements[1].color = (0, 0, 0, 1)
        L.new(vor.outputs["Distance"], ramp.inputs["Fac"])
        # Vary brightness per star.
        mul = N.new("ShaderNodeMix")
        mul.data_type = "RGBA"
        mul.blend_type = "MULTIPLY"
        mul.inputs["Factor"].default_value = 1
        L.new(ramp.outputs["Color"], mul.inputs["A"])
        L.new(vor.outputs["Color"], mul.inputs["B"])
        if total is None:
            total = mul.outputs["Result"]
        else:
            add = N.new("ShaderNodeMix")
            add.data_type = "RGBA"
            add.blend_type = "ADD"
            add.inputs["Factor"].default_value = 1
            L.new(total, add.inputs["A"])
            L.new(mul.outputs["Result"], add.inputs["B"])
            total = add.outputs["Result"]
    # Faint rust-and-teal nebula, like v1's.
    noise = N.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 1.6
    noise.inputs["Detail"].default_value = 6
    L.new(coord.outputs["Generated"], noise.inputs["Vector"])
    nr = N.new("ShaderNodeValToRGB")
    els = nr.color_ramp.elements
    els[0].position = 0.42
    els[0].color = (0, 0, 0, 1)
    els[1].position = 0.75
    els[1].color = (0.05 * nebula, 0.016 * nebula, 0.012 * nebula, 1)
    e = els.new(0.62)
    e.color = (0.006 * nebula, 0.016 * nebula, 0.03 * nebula, 1)
    L.new(noise.outputs["Fac"], nr.inputs["Fac"])
    add = N.new("ShaderNodeMix")
    add.data_type = "RGBA"
    add.blend_type = "ADD"
    add.inputs["Factor"].default_value = 1
    L.new(total, add.inputs["A"])
    L.new(nr.outputs["Color"], add.inputs["B"])
    bg = N.new("ShaderNodeBackground")
    L.new(add.outputs["Result"], bg.inputs["Color"])
    # Barely-there ambient so the dark side is not pure black.
    lp = N.new("ShaderNodeLightPath")
    amb = N.new("ShaderNodeBackground")
    amb.inputs["Color"].default_value = (0.012, 0.014, 0.022, 1)
    mix = N.new("ShaderNodeMixShader")
    L.new(lp.outputs["Is Camera Ray"], mix.inputs["Fac"])
    L.new(amb.outputs[0], mix.inputs[1])
    L.new(bg.outputs[0], mix.inputs[2])
    L.new(mix.outputs[0], out.inputs["Surface"])


def world_dusk():
    """Sunset toward +Y: orange at the horizon, mauve, then deep blue; faint stars; warm haze."""
    w = bpy.context.scene.world
    nt, N, L = nodes_of(w)
    out = N.new("ShaderNodeOutputWorld")
    coord = N.new("ShaderNodeTexCoord")
    sep = N.new("ShaderNodeSeparateXYZ")
    norm = N.new("ShaderNodeVectorMath")
    norm.operation = "NORMALIZE"
    L.new(coord.outputs["Generated"], norm.inputs[0])
    L.new(norm.outputs[0], sep.inputs[0])
    sky = N.new("ShaderNodeValToRGB")
    e = sky.color_ramp.elements
    e[0].position = 0.0
    e[0].color = (0.9, 0.32, 0.08, 1)
    e[1].position = 0.6
    e[1].color = (0.004, 0.008, 0.03, 1)
    for pos, col in ((0.05, (0.5, 0.16, 0.14, 1)), (0.12, (0.12, 0.06, 0.16, 1)), (0.25, (0.025, 0.035, 0.1, 1))):
        n = e.new(pos)
        n.color = col
    L.new(sep.outputs["Z"], sky.inputs["Fac"])
    # Extra warmth toward the sun (+Y), hugging the horizon.
    yclamp = N.new("ShaderNodeMath")
    yclamp.operation = "MAXIMUM"
    yclamp.inputs[1].default_value = 0
    L.new(sep.outputs["Y"], yclamp.inputs[0])
    ypow = N.new("ShaderNodeMath")
    ypow.operation = "POWER"
    ypow.inputs[1].default_value = 3
    L.new(yclamp.outputs[0], ypow.inputs[0])
    zfall = N.new("ShaderNodeMapRange")
    zfall.inputs["From Min"].default_value = 0.0
    zfall.inputs["From Max"].default_value = 0.28
    zfall.inputs["To Min"].default_value = 1
    zfall.inputs["To Max"].default_value = 0
    L.new(sep.outputs["Z"], zfall.inputs["Value"])
    warm = N.new("ShaderNodeMath")
    warm.operation = "MULTIPLY"
    L.new(ypow.outputs[0], warm.inputs[0])
    L.new(zfall.outputs[0], warm.inputs[1])
    addw = N.new("ShaderNodeMix")
    addw.data_type = "RGBA"
    addw.blend_type = "ADD"
    addw.inputs["B"].default_value = (1.2, 0.48, 0.14, 1)
    L.new(warm.outputs[0], addw.inputs["Factor"])
    L.new(sky.outputs["Color"], addw.inputs["A"])
    # Faint stars high up.
    vor = N.new("ShaderNodeTexVoronoi")
    vor.inputs["Scale"].default_value = 220
    L.new(norm.outputs[0], vor.inputs["Vector"])
    st = N.new("ShaderNodeValToRGB")
    st.color_ramp.elements[0].color = (0.9, 0.9, 1.0, 1)
    st.color_ramp.elements[1].position = 0.03
    st.color_ramp.elements[1].color = (0, 0, 0, 1)
    L.new(vor.outputs["Distance"], st.inputs["Fac"])
    zmask = N.new("ShaderNodeMapRange")
    zmask.inputs["From Min"].default_value = 0.15
    zmask.inputs["From Max"].default_value = 0.5
    L.new(sep.outputs["Z"], zmask.inputs["Value"])
    smul = N.new("ShaderNodeMix")
    smul.data_type = "RGBA"
    smul.blend_type = "MULTIPLY"
    smul.inputs["Factor"].default_value = 1
    L.new(st.outputs["Color"], smul.inputs["A"])
    L.new(zmask.outputs["Result"], smul.inputs["B"])
    adds = N.new("ShaderNodeMix")
    adds.data_type = "RGBA"
    adds.blend_type = "ADD"
    adds.inputs["Factor"].default_value = 1
    L.new(addw.outputs["Result"], adds.inputs["A"])
    L.new(smul.outputs["Result"], adds.inputs["B"])
    bg = N.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = 1.0
    L.new(adds.outputs["Result"], bg.inputs["Color"])
    # The sky lights the scene only faintly, so the sun and lamps make the toon bands.
    amb = N.new("ShaderNodeBackground")
    amb.inputs["Color"].default_value = (0.05, 0.045, 0.07, 1)
    lp = N.new("ShaderNodeLightPath")
    mix = N.new("ShaderNodeMixShader")
    L.new(lp.outputs["Is Camera Ray"], mix.inputs["Fac"])
    L.new(amb.outputs[0], mix.inputs[1])
    L.new(bg.outputs[0], mix.inputs[2])
    L.new(mix.outputs[0], out.inputs["Surface"])


def haze(size=300, density=0.0005):
    """Warm dusk haze in a bounded box (a world volume would swallow the sky)."""
    mat = bpy.data.materials.new("haze")
    nt, N, L = nodes_of(mat)
    out = N.new("ShaderNodeOutputMaterial")
    vol = N.new("ShaderNodeVolumePrincipled")
    vol.inputs["Color"].default_value = (1.0, 0.7, 0.5, 1)
    vol.inputs["Density"].default_value = density
    vol.inputs["Anisotropy"].default_value = 0.6
    L.new(vol.outputs[0], out.inputs["Volume"])
    return prim("cube", "haze", loc=(0, 0, size * 0.1), scale=(size, size, size * 0.1 + 30), mat=mat, size=2, smooth=False)


# ---------------------------------------------------------------- render setup


def setup_render(sc, frames, samples=32, kuwahara=9, glare=True):
    try:
        sc.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        sc.render.engine = "BLENDER_EEVEE"
    sc.render.resolution_x, sc.render.resolution_y = RX, RY
    sc.render.resolution_percentage = 100
    sc.render.fps = FPS
    sc.frame_start, sc.frame_end = 1, frames
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_depth = "8"
    sc.render.image_settings.color_mode = "RGB"
    sc.render.film_transparent = False
    ee = sc.eevee
    ee.taa_render_samples = samples
    for attr, val in (("use_volumetric_shadows", True), ("volumetric_end", 900.0),
                      ("volumetric_tile_size", "4"), ("volumetric_samples", 64),
                      ("use_shadows", True), ("use_raytracing", False)):
        if hasattr(ee, attr):
            try:
                setattr(ee, attr, val)
            except Exception:
                pass
    sc.render.use_motion_blur = True
    sc.render.motion_blur_shutter = 0.5
    sc.view_settings.view_transform = "AgX"
    try:
        sc.view_settings.look = "AgX - Punchy"
    except TypeError:
        pass
    # Compositor: glow, then the painterly Kuwahara pass.
    sc.use_nodes = True
    nt = sc.node_tree
    nt.nodes.clear()
    rl = nt.nodes.new("CompositorNodeRLayers")
    comp = nt.nodes.new("CompositorNodeComposite")
    sock = rl.outputs["Image"]
    if glare:
        g = nt.nodes.new("CompositorNodeGlare")
        for t in ("BLOOM", "FOG_GLOW"):
            try:
                g.glare_type = t
                break
            except TypeError:
                continue
        for name, val in (("threshold", 1.0), ("size", 8), ("mix", -0.55), ("quality", "HIGH")):
            if hasattr(g, name):
                setattr(g, name, val)
        for name, val in (("Threshold", 1.0), ("Strength", 0.55), ("Size", 0.7)):
            if name in g.inputs:
                g.inputs[name].default_value = val
        nt.links.new(sock, g.inputs["Image"])
        sock = g.outputs["Image"]
    if kuwahara:
        k = nt.nodes.new("CompositorNodeKuwahara")
        k.variation = "ANISOTROPIC"
        if "Size" in k.inputs:
            k.inputs["Size"].default_value = kuwahara
        elif hasattr(k, "size"):
            k.size = kuwahara
        for name, val in (("Sharpness", 0.55), ("Eccentricity", 1.0), ("Uniformity", 4)):
            if name in k.inputs:
                k.inputs[name].default_value = val
            elif hasattr(k, name.lower()):
                setattr(k, name.lower(), val)
        nt.links.new(sock, k.inputs["Image"])
        sock = k.outputs["Image"]
    nt.links.new(sock, comp.inputs["Image"])
    try:
        sc.render.compositor_device = "GPU"
    except (AttributeError, TypeError):
        pass


def camera(name, lens=35):
    cam = bpy.data.cameras.new(name)
    cam.lens = lens
    cam.sensor_fit = "HORIZONTAL"
    cam.sensor_width = 36
    cam.clip_start = 0.1
    cam.clip_end = 8000
    o = link(bpy.data.objects.new(name, cam))
    bpy.context.scene.camera = o
    return o


def aim(cam_obj, target):
    """A Track To target, so camera moves are just two paths."""
    t = empty(cam_obj.name + "_target", loc=target)
    c = cam_obj.constraints.new("TRACK_TO")
    c.target = t
    c.track_axis = "TRACK_NEGATIVE_Z"
    c.up_axis = "UP_Y"
    return t


def dust(name, n, radius, center=(0, 0, 0), size=0.06, strength=3.0, seed=5, axis_tunnel=False):
    """Many tiny emissive specks as one mesh (parallax for the camera)."""
    rnd = random.Random(seed)
    me = bpy.data.meshes.new(name)
    verts, faces = [], []
    for _ in range(n):
        if axis_tunnel:
            a = rnd.uniform(0, 2 * math.pi)
            r = rnd.uniform(radius * 0.08, radius)
            p = Vector((rnd.uniform(-radius * 0.3, radius * 4), math.cos(a) * r, math.sin(a) * r))
        else:
            p = Vector((rnd.uniform(-1, 1), rnd.uniform(-1, 1), rnd.uniform(-1, 1))) * radius
        s = size * rnd.uniform(0.5, 1.5)
        b = len(verts)
        verts += [p + Vector(v) * s for v in ((1, 1, 1), (-1, -1, 1), (-1, 1, -1), (1, -1, -1))]
        faces += [(b, b + 1, b + 2), (b, b + 3, b + 1), (b, b + 2, b + 3), (b + 1, b + 3, b + 2)]
    me.from_pydata(verts, [], faces)
    o = link(bpy.data.objects.new(name, me))
    o.location = center
    o.data.materials.append(glow(name, (0.85, 0.9, 1.0), strength))
    o.visible_shadow = False
    return o


# ---------------------------------------------------------------- the shots


def sun_toward(name, direction, energy, color):
    """A sun light whose rays travel along `direction`."""
    l = bpy.data.lights.new(name, "SUN")
    l.energy = energy
    l.color = color
    o = link(bpy.data.objects.new(name, l))
    o.rotation_mode = "QUATERNION"
    o.rotation_quaternion = Vector(direction).to_track_quat("-Z", "Y")
    return o


def engine_level(parts, frames_core, frames_halo, frames_light):
    core = parts["engine_mat"].node_tree.nodes["Emission"].inputs["Strength"]
    halo_em = parts["halo"].active_material.node_tree.nodes["Emission"].inputs["Strength"]
    el = parts["engine_light"].data
    for f, v in frames_core:
        core.default_value = v
        core.keyframe_insert("default_value", frame=f)
    for f, v in frames_halo:
        halo_em.default_value = v
        halo_em.keyframe_insert("default_value", frame=f)
    for f, v in frames_light:
        el.energy = v
        el.keyframe_insert("energy", frame=f)


def shot_cold(sc):
    """0-10.5 s: the ship crosses close in front of the camera at an angle, between the two
    title lines (nearest at ~5 s), then drifts away to the right, engine toward us."""
    world_space(nebula=1.0)
    root, parts = build_ship()
    gear(parts, False)  # retracted in space
    sun_toward("starlight", (1.0, 0.45, -0.7), 3.2, (1.0, 0.9, 0.78))
    sun_toward("rim", (-0.6, -0.3, -0.3), 1.2, (0.55, 0.65, 1.0))
    cam = camera("cam_cold", lens=32)
    tgt = aim(cam, (100, 0, 14))
    key(cam, "location", 1, (0, 0, 0))
    key(cam, "location", 252, (3, -1.5, 0.6))
    key(tgt, "location", 1, (100, 18, 17))
    key(tgt, "location", 125, (100, 24, 18))
    key(tgt, "location", 252, (100, -18, 12))
    dust("dust", 1500, 70, center=(45, 0, 6), size=0.018, strength=6.0)
    # The path: a straight line at -35 degrees yaw; s metres along it.
    yaw = math.radians(-35)
    d = Vector((math.cos(yaw), math.sin(yaw), 0))
    p0 = Vector((-12, 47, 5.0))
    for f, s_ in ((1, -18), (125, 63), (252, 168)):
        key(root, "location", f, p0 + d * s_ + Vector((0, 0, 0.012 * s_)))
    key(root, "rotation_euler", 1, (math.radians(-9), math.radians(-2), yaw))
    key(root, "rotation_euler", 125, (math.radians(-4), math.radians(-1), yaw))
    key(root, "rotation_euler", 252, (math.radians(3), math.radians(0), yaw + math.radians(2)))
    # Engine surges on the 6.0 s thud (frame 145) and the 8.6 s one (frame 207).
    engine_level(parts,
                 ((1, 22), (143, 22), (147, 60), (175, 28), (205, 28), (209, 55), (240, 30)),
                 ((1, 2.2), (143, 2.2), (147, 6.0), (175, 2.8), (205, 2.8), (209, 5.0), (240, 3.0)),
                 ((1, 30000), (143, 30000), (147, 90000), (175, 40000), (205, 40000), (209, 80000), (240, 45000)))
    smooth_keys(root, tgt, cam)
    setup_render(sc, SHOTS["cold"], samples=32, kuwahara=4)


def outpost():
    rnd = random.Random(11)
    sand = toon("sand", (0.62, 0.38, 0.22), grime=0.25, ambient=0.3, soft=True)
    white = toon("adobe", (0.78, 0.72, 0.62), grime=0.45, ambient=0.12)
    rock = toon("rock", (0.36, 0.2, 0.2), grime=0.3, ambient=0.14)
    concrete = toon("pad", (0.42, 0.4, 0.38), grime=0.5, ambient=0.12)
    warm = glow("lamp", (1.0, 0.66, 0.32), 10)
    red = glow("beacon", (1.0, 0.1, 0.05), 0)
    # Ground with soft dunes.
    g = prim("plane", "ground", mat=sand, size=3000)
    sub = g.modifiers.new("sub", "SUBSURF")
    sub.subdivision_type = "SIMPLE"
    sub.levels = sub.render_levels = 6
    tex = bpy.data.textures.new("dunes", "CLOUDS")
    tex.noise_scale = 120
    disp = g.modifiers.new("disp", "DISPLACE")
    disp.texture = tex
    disp.strength = 2.5
    disp.mid_level = 0.62
    # Pad: flat, with rim lamps.
    prim("cyl", "pad", loc=(0, 0, 0.6), mat=concrete, radius=17, depth=1.2, vertices=64, bevel=0.2)
    prim("cyl", "pad_ring", loc=(0, 0, 1.21), mat=toon("pad_mark", (0.55, 0.42, 0.2), grime=0.6), radius=12,
         depth=0.02, vertices=64)
    prim("cyl", "pad_inner", loc=(0, 0, 1.22), mat=concrete, radius=11.4, depth=0.02, vertices=64)
    for i in range(16):
        a = i / 16 * 2 * math.pi
        prim("sphere", f"padlamp{i}", loc=(16.4 * math.cos(a), 16.4 * math.sin(a), 1.3), mat=warm,
             radius=0.22, segments=8, ring_count=6)
    # Domes, behind the pad toward the sunset.
    for i, (x, y, r) in enumerate([(-42, 46, 10), (-64, 26, 6.5), (-24, 74, 7.5), (-78, 64, 8)]):
        prim("sphere", f"dome{i}", loc=(x, y, -r * 0.05), scale=(1, 1, 0.82), mat=white, radius=r,
             segments=40, ring_count=20)
        for j in range(rnd.randint(3, 5)):
            a = rnd.uniform(-2.6, -0.6)  # windows on the side facing the camera
            el = rnd.uniform(0.15, 0.5)
            p = Vector((x + r * math.cos(a) * math.cos(el), y + r * math.sin(a) * math.cos(el),
                        -r * 0.05 + r * 0.82 * math.sin(el)))
            prim("cube", f"dome{i}_win{j}", loc=p, scale=(0.5, 0.5, 0.35), mat=warm, size=1)
        prim("cube", f"dome{i}_door", loc=(x + r * math.cos(-1.6) * 0.98, y + r * math.sin(-1.6) * 0.98, 1.0),
             scale=(1.2, 0.5, 2.0), mat=warm, size=1)
    # Tower with the red beacon.
    tx, ty = -54, 50
    prim("cyl", "tower", loc=(tx, ty, 14), mat=white, radius=2.6, depth=28, vertices=24)
    prim("cyl", "tower_cab", loc=(tx, ty, 29), mat=white, radius=4.2, depth=3.6, vertices=24, bevel=0.2)
    prim("cyl", "tower_glass", loc=(tx, ty, 29.2), mat=warm, radius=4.25, depth=1.0, vertices=24)
    prim("cyl", "antenna", loc=(tx, ty, 35), mat=toon("mast", (0.3, 0.3, 0.32)), radius=0.25, depth=8, vertices=8)
    beacon = prim("sphere", "beacon", loc=(tx, ty, 39.3), mat=red, radius=0.7)
    bl = bpy.data.lights.new("beacon_light", "POINT")
    bl.color = (1.0, 0.1, 0.05)
    bl.shadow_soft_size = 0.5
    blo = link(bpy.data.objects.new("beacon_light", bl))
    blo.location = (tx, ty, 39.3)
    em = red.node_tree.nodes["Emission"]
    for f in range(1, 260, 2):
        on = 1.0 if (f % 36) < 12 else 0.0
        em.inputs["Strength"].default_value = 40 * on
        em.inputs["Strength"].keyframe_insert("default_value", frame=f)
        bl.energy = 20000 * on
        bl.keyframe_insert("energy", frame=f)
    # Shacks, tanks, a fence of poles: a working outpost.
    for i in range(10):
        x, y = rnd.uniform(-90, -18), rnd.uniform(10, 95)
        if math.hypot(x, y) < 22:
            continue
        if rnd.random() < 0.5:
            prim("cube", f"shack{i}", loc=(x, y, 1.6), scale=(rnd.uniform(2, 5), rnd.uniform(2, 4), 1.6),
                 rot=(0, 0, rnd.uniform(0, 3)), mat=white, size=2, bevel=0.15)
        else:
            prim("cyl", f"tank{i}", loc=(x, y, 2.2), rot=(math.pi / 2, 0, rnd.uniform(0, 3)), mat=rock,
                 radius=1.6, depth=6, vertices=20)
    # Mesas on the horizon.
    far = toon("mesa", (0.22, 0.12, 0.16), grime=0.2, ambient=0.25)
    for i in range(22):
        a = math.radians(rnd.uniform(20, 160))
        d = rnd.uniform(1300, 2000)
        w = rnd.uniform(50, 220)
        h = rnd.uniform(18, 55)
        m = prim("cube", f"mesa{i}", loc=(d * math.cos(a), d * math.sin(a), h / 2 - 6),
                 scale=(w, rnd.uniform(40, 120), h / 2), rot=(0, 0, a + rnd.uniform(-0.4, 0.4)), mat=far,
                 size=2, bevel=6)
        # A sloped skirt so they read as eroded mesas, not boxes.
        prim("cone", f"mesa{i}_skirt", loc=(d * math.cos(a), d * math.sin(a), h * 0.18 - 6),
             scale=(w * 1.35, 80, h * 0.36), rot=(0, 0, a), mat=far, radius1=1, radius2=0.6, depth=1, vertices=7)
    # Two moons in the dusk sky.
    moon = glow("moon", (0.95, 0.9, 0.78), 2.2)
    moon2 = glow("moon2", (0.95, 0.72, 0.68), 1.6)
    for name, d, r, m in (("moon_big", Vector((-0.34, 1, 0.21)), 150, moon), ("moon_small", Vector((-0.08, 1, 0.13)), 55, moon2)):
        prim("sphere", name, loc=d.normalized() * 2400, mat=m, radius=r, segments=48, ring_count=24)
    return beacon


def shot_arrival(sc):
    """9.2-18.6 s: crane down over the outpost at dusk; the ship descends and settles on the pad,
    the ramp drops and the crew stand in the hatch light."""
    world_dusk()
    outpost()
    haze()
    sun = bpy.data.lights.new("sun", "SUN")
    sun.energy = 3.0
    sun.color = (1.0, 0.55, 0.28)
    sun.angle = math.radians(2)
    so = link(bpy.data.objects.new("sun", sun))
    so.rotation_euler = (math.radians(84), 0, math.radians(180))  # low, from +Y toward the camera
    fill = bpy.data.lights.new("moonfill", "SUN")
    fill.energy = 0.35
    fill.color = (0.45, 0.55, 1.0)
    fo = link(bpy.data.objects.new("moonfill", fill))
    fo.rotation_euler = (math.radians(40), 0, math.radians(-30))
    root, parts = build_ship(engine_strength=0.6)
    pad_top = 1.23
    land_z = pad_top + 5.1 - 0.05
    # Descent: from high and behind-left, slowing into the pad.
    key(root, "location", 1, (-95, 38, 62))
    key(root, "location", 70, (-38, 14, 30))
    key(root, "location", 130, (-4, 1.5, land_z + 2.5))
    key(root, "location", 156, (0, 0, land_z))
    key(root, "location", 226, (0, 0, land_z))
    key(root, "rotation_euler", 1, (math.radians(6), math.radians(-8), math.radians(-28)))
    key(root, "rotation_euler", 130, (math.radians(0), math.radians(2), math.radians(-6)))
    key(root, "rotation_euler", 156, (0, 0, math.radians(-4)))
    key(root, "rotation_euler", 226, (0, 0, math.radians(-4)))
    st = parts["struts"]
    gear(parts, False, frame=1)
    gear(parts, True, frame=88)
    key(st, "scale", 88, (1, 1, 0.08))
    key(st, "scale", 128, (1, 1, 1.0))
    # Engine: strong on the way in, idles down after touchdown.
    el = parts["engine_light"].data
    for f, e in ((1, 50000), (150, 40000), (180, 6000), (226, 4000)):
        el.energy = e
        el.keyframe_insert("energy", frame=f)
    core = parts["engine_mat"].node_tree.nodes["Emission"].inputs["Strength"]
    for f, v in ((1, 40), (150, 30), (185, 4), (226, 3)):
        core.default_value = v
        core.keyframe_insert("default_value", frame=f)
    halo_em = parts["halo"].active_material.node_tree.nodes["Emission"].inputs["Strength"]
    for f, v in ((1, 9), (150, 7), (185, 0.6), (226, 0.4)):
        halo_em.default_value = v
        halo_em.keyframe_insert("default_value", frame=f)
    # Ramp drops at 16.0 s (frame 166); hatch light comes up; the crew stand in it.
    ramp = parts["ramp"]
    key(ramp, "rotation_euler", 1, (0, 0, 0))
    key(ramp, "rotation_euler", 164, (0, 0, 0))
    key(ramp, "rotation_euler", 196, (0, math.radians(46), 0))
    hatch = parts["hatch"].active_material.node_tree.nodes["Emission"].inputs["Strength"]
    for f, v in ((1, 0), (164, 0), (180, 14), (226, 14)):
        hatch.default_value = v
        hatch.keyframe_insert("default_value", frame=f)
    hl = bpy.data.lights.new("hatch_spill", "AREA")
    hl.color = (1.0, 0.7, 0.42)
    hl.size = 3
    hlo = link(bpy.data.objects.new("hatch_spill", hl))
    hlo.parent = root
    hlo.location = (10.8, 0, -2.1)
    for f, v in ((1, 0), (164, 0), (184, 9000), (226, 9000)):
        hl.energy = v
        hl.keyframe_insert("energy", frame=f)
    crew_root = empty("crew", parent=root)
    crew(crew_root, (13.2, 0, -5.05))
    key(crew_root, "scale", 1, (0, 0, 0))
    key(crew_root, "scale", 176, (0, 0, 0))
    key(crew_root, "scale", 177, (1, 1, 1))
    for fc in crew_root.animation_data.action.fcurves if hasattr(crew_root.animation_data.action, "fcurves") else []:
        for k in fc.keyframe_points:
            k.interpolation = "CONSTANT"
    # Camera: a slow crane from high and wide down to the pad, looking toward the sunset.
    cam = camera("cam_arrival", lens=30)
    key(cam, "location", 1, (58, -120, 52))
    key(cam, "location", 160, (34, -62, 12))
    key(cam, "location", 226, (24, -38, 5.0))
    tgt = aim(cam, (0, 0, 0))
    key(tgt, "location", 1, (-40, 30, 26))
    key(tgt, "location", 150, (-4, 6, 6))
    key(tgt, "location", 226, (3, 2, 4.2))
    smooth_keys(root, cam, tgt, st, ramp)
    setup_render(sc, SHOTS["arrival"], samples=48)


def shot_jump(sc):
    """18.6-19.85 s: from behind, the engine flares, the stars stretch and the ship is gone."""
    world_space(nebula=0.5)
    root, parts = build_ship()
    gear(parts, False)  # retracted in space
    kl = link(bpy.data.objects.new("starlight", bpy.data.lights.new("starlight", "SUN")))
    kl.data.energy = 2.0
    kl.rotation_euler = (math.radians(-55), math.radians(-30), math.radians(30))
    cam = camera("cam_jump", lens=40)
    key(cam, "location", 1, (-46, -11, 7))
    key(cam, "location", 30, (-44, -10.5, 6.6))
    tgt = aim(cam, (0, 0, 1))
    streaks = empty("streaks", loc=(-30, 0, 0))
    d = dust("streak_dust", 3000, 160, center=(30, 0, 0), size=0.08, strength=6.0, seed=9, axis_tunnel=True)
    d.parent = streaks
    key(streaks, "scale", 1, (1, 1, 1))
    key(streaks, "scale", 8, (1, 1, 1))
    key(streaks, "scale", 22, (90, 1, 1))
    key(streaks, "scale", 30, (140, 1, 1))
    key(root, "location", 1, (0, 0, 0))
    key(root, "location", 9, (3, 0, 0))
    key(root, "location", 16, (40, 0, 0.4))
    key(root, "location", 22, (900, 0, 3))
    key(root, "location", 30, (4000, 0, 6))
    el = parts["engine_light"].data
    for f, e in ((1, 60000), (8, 250000), (20, 500000), (30, 500000)):
        el.energy = e
        el.keyframe_insert("energy", frame=f)
    core = parts["engine_mat"].node_tree.nodes["Emission"].inputs["Strength"]
    for f, v in ((1, 40), (8, 120), (22, 300)):
        core.default_value = v
        core.keyframe_insert("default_value", frame=f)
    smooth_keys(cam)
    setup_render(sc, SHOTS["jump"], samples=24)
    sc.render.motion_blur_shutter = 1.0


def shot_wide(sc):
    """52.5-60 s: under the date card, the ship small, heading for the desert world's limb."""
    world_space(nebula=1.0)
    D, R = 114000.0, 98700.0
    centre = Vector((0, math.cos(math.radians(68)), -math.sin(math.radians(68)))) * D
    planet_mat = toon("planet", (0.72, 0.42, 0.22), grime=0.7, ambient=0.02, soft=True)
    prim("sphere", "planet", loc=centre, mat=planet_mat, radius=R, segments=256, ring_count=128)
    atm = prim("sphere", "atmo", loc=centre, mat=halo("atmo", (1.0, 0.6, 0.35), 6.0, power=4), radius=R * 1.012,
               segments=256, ring_count=128)
    atm.visible_shadow = False
    sun_toward("sun", (0.3, -1.0, -0.12), 6.0, (1.0, 0.8, 0.58))  # from behind the planet: a crescent
    root, parts = build_ship()
    gear(parts, False)  # retracted in space
    cam = camera("cam_wide", lens=35)
    cam.location = (0, 0, 0)
    tgt = aim(cam, (0, 1000, 20))
    key(tgt, "location", 1, (0, 1000, 25))
    key(tgt, "location", 180, (0, 1000, 15))
    key(root, "location", 1, (95, 300, 46))
    key(root, "location", 180, (78, 345, 41))
    for f in (1, 180):
        key(root, "rotation_euler", f, (math.radians(-4), 0, math.radians(97 + (f - 1) * 0.01)))
    for fc in root.animation_data.action.fcurves if hasattr(root.animation_data.action, "fcurves") else []:
        for k in fc.keyframe_points:
            k.interpolation = "LINEAR"
    engine_level(parts, ((1, 30),), ((1, 3.0),), ((1, 40000),))
    setup_render(sc, SHOTS["wide"], samples=32, kuwahara=4)
    cam.data.clip_end = 300000


BUILDERS = {"cold": shot_cold, "arrival": shot_arrival, "jump": shot_jump, "wide": shot_wide}


def build(shot):
    sc = reset()
    BUILDERS[shot](sc)
    sc.frame_set(1)
    return sc


def render(shot, first=None, last=None, still=None):
    sc = build(shot)
    if still is not None:
        sc.frame_set(still)
        sc.render.filepath = os.path.join(TEASER_DIR, "shots", f"still-{shot}.png")
        bpy.ops.render.render(write_still=True)
        return
    out = os.path.join(TEASER_DIR, "shots", shot)
    os.makedirs(out, exist_ok=True)
    sc.frame_start = first or 1
    sc.frame_end = last or SHOTS[shot]
    sc.render.filepath = os.path.join(out, "f")
    sc.render.use_overwrite = False
    sc.render.use_placeholder = True
    bpy.ops.render.render(animation=True)


if __name__ == "__main__" and "--" in sys.argv:
    args = sys.argv[sys.argv.index("--") + 1:]
    if args[0] == "render":
        render(args[1], *(int(a) for a in args[2:4]))
    elif args[0] == "still":
        render(args[1], still=int(args[2]))
