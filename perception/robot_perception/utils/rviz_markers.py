"""Build visualization_msgs/MarkerArray for detection bbox RViz display."""
import numpy as np
from geometry_msgs.msg import Point, Pose, Vector3
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


def bgr_to_color_rgba(bgr, alpha=1.0):
    b, g, r = bgr
    c = ColorRGBA()
    c.r = float(r) / 255.0
    c.g = float(g) / 255.0
    c.b = float(b) / 255.0
    c.a = float(alpha)
    return c


def _aabb_corners(aabb):
    pmin = aabb['min']
    pmax = aabb['max']
    return [
        (float(pmin[0]), float(pmin[1]), float(pmin[2])),
        (float(pmax[0]), float(pmin[1]), float(pmin[2])),
        (float(pmax[0]), float(pmax[1]), float(pmin[2])),
        (float(pmin[0]), float(pmax[1]), float(pmin[2])),
        (float(pmin[0]), float(pmin[1]), float(pmax[2])),
        (float(pmax[0]), float(pmin[1]), float(pmax[2])),
        (float(pmax[0]), float(pmax[1]), float(pmax[2])),
        (float(pmin[0]), float(pmax[1]), float(pmax[2])),
    ]


_AABB_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def _make_point(x, y, z):
    p = Point()
    p.x = x
    p.y = y
    p.z = z
    return p


def make_aabb_line_marker(marker_id, frame_id, stamp, aabb, color_bgr,
                          line_width=0.002, ns='bbox3d'):
    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp = stamp
    m.ns = ns
    m.id = marker_id
    m.type = Marker.LINE_LIST
    m.action = Marker.ADD
    m.scale.x = line_width
    m.color = bgr_to_color_rgba(color_bgr, alpha=1.0)
    m.pose.orientation.w = 1.0

    corners = _aabb_corners(aabb)
    for i, j in _AABB_EDGES:
        c0, c1 = corners[i], corners[j]
        m.points.append(_make_point(*c0))
        m.points.append(_make_point(*c1))
    return m


def make_aabb_cube_marker(marker_id, frame_id, stamp, aabb, color_bgr,
                          alpha=0.15, ns='bbox3d_fill'):
    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp = stamp
    m.ns = ns
    m.id = marker_id
    m.type = Marker.CUBE
    m.action = Marker.ADD
    m.pose = Pose()
    m.pose.position.x = float(aabb['center'][0])
    m.pose.position.y = float(aabb['center'][1])
    m.pose.position.z = float(aabb['center'][2])
    m.pose.orientation.w = 1.0
    m.scale = Vector3(
        x=max(float(aabb['size'][0]), 1e-4),
        y=max(float(aabb['size'][1]), 1e-4),
        z=max(float(aabb['size'][2]), 1e-4),
    )
    m.color = bgr_to_color_rgba(color_bgr, alpha=alpha)
    return m


def make_text_marker(marker_id, frame_id, stamp, aabb, text, color_bgr,
                     text_height=0.015, ns='bbox3d_label'):
    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp = stamp
    m.ns = ns
    m.id = marker_id
    m.type = Marker.TEXT_VIEW_FACING
    m.action = Marker.ADD
    m.pose.position.x = float(aabb['center'][0])
    m.pose.position.y = float(aabb['center'][1])
    m.pose.position.z = float(aabb['max'][2] + text_height)
    m.pose.orientation.w = 1.0
    m.scale.z = text_height
    m.color = bgr_to_color_rgba(color_bgr, alpha=1.0)
    m.text = text
    return m


def make_delete_all_marker(stamp, frame_id):
    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp = stamp
    m.action = Marker.DELETEALL
    return m


def _append_aabb_markers(markers, results, stamp, frame_id, aabb_key, args,
                         id_offset=0):
    marker_id = id_offset
    for r in results:
        aabb = r.get(aabb_key)
        if aabb is None:
            continue
        label = r['tag']
        inst = r['instance_id']
        color = r['color']
        text = f"{label} #{inst} ({r['score']:.2f})"
        ns_base = f'{frame_id}/{label}_{inst}'

        markers.markers.append(make_aabb_line_marker(
            marker_id, frame_id, stamp, aabb, color, ns=f'{ns_base}/wire'))
        marker_id += 1

        if args.rviz_show_fill:
            markers.markers.append(make_aabb_cube_marker(
                marker_id, frame_id, stamp, aabb, color,
                alpha=args.rviz_fill_alpha, ns=f'{ns_base}/fill'))
            marker_id += 1

        markers.markers.append(make_text_marker(
            marker_id, frame_id, stamp, aabb, text, color, ns=f'{ns_base}/label'))
        marker_id += 1
    return marker_id


def build_camera_marker_array(results, stamp, frame_id, args):
    markers = MarkerArray()
    if not results or not args.publish_camera_bbox:
        markers.markers.append(make_delete_all_marker(stamp, frame_id))
        return markers
    _append_aabb_markers(markers, results, stamp, frame_id, 'aabb_cam', args)
    return markers


def build_workbench_marker_array(results, stamp, frame_id, args):
    markers = MarkerArray()
    if not results or not args.publish_workbench_bbox:
        markers.markers.append(make_delete_all_marker(stamp, frame_id))
        return markers
    work_results = [r for r in results if r.get('aabb_work') is not None]
    if not work_results:
        markers.markers.append(make_delete_all_marker(stamp, frame_id))
        return markers
    _append_aabb_markers(markers, work_results, stamp, frame_id, 'aabb_work', args)
    return markers


def make_workbench_plane_marker(stamp, frame_id, plane_state, size_m, thickness_m=0.004):
    """Semi-transparent CUBE aligned with the estimated workbench plane."""
    from scipy.spatial.transform import Rotation

    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp = stamp
    m.ns = 'workbench_plane'
    m.id = 0
    m.type = Marker.CUBE
    m.action = Marker.ADD
    m.pose.position.x = float(plane_state.centroid[0])
    m.pose.position.y = float(plane_state.centroid[1])
    m.pose.position.z = float(plane_state.centroid[2])

    normal = np.asarray(plane_state.normal, dtype=np.float64)
    rot, _ = Rotation.align_vectors([normal], [[0.0, 0.0, 1.0]])
    q = rot.as_quat()
    m.pose.orientation.x = float(q[0])
    m.pose.orientation.y = float(q[1])
    m.pose.orientation.z = float(q[2])
    m.pose.orientation.w = float(q[3])

    m.scale = Vector3(
        x=max(float(size_m), 0.05),
        y=max(float(size_m), 0.05),
        z=max(float(thickness_m), 0.001),
    )
    c = ColorRGBA()
    c.r = 0.15
    c.g = 0.85
    c.b = 0.25
    c.a = 0.35
    m.color = c
    return m


def build_workbench_plane_marker_array(stamp, frame_id, plane_state, args):
    markers = MarkerArray()
    if not getattr(args, 'publish_workbench_plane_marker', True):
        markers.markers.append(make_delete_all_marker(stamp, frame_id))
        return markers
    if plane_state is None or not getattr(plane_state, 'valid', False):
        markers.markers.append(make_delete_all_marker(stamp, frame_id))
        return markers
    size_m = float(getattr(args, 'workbench_plane_marker_size_m', 0.8))
    thickness_m = float(getattr(args, 'workbench_plane_marker_thickness_m', 0.004))
    markers.markers.append(
        make_workbench_plane_marker(stamp, frame_id, plane_state, size_m, thickness_m))
    return markers


def make_hybrid_surface_marker(marker_id, frame_id, stamp, mesh, color_bgr,
                               alpha=0.55, ns='hybrid_surface'):
    """TRIANGLE_LIST marker from hybrid surface mesh dict."""
    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp = stamp
    m.ns = ns
    m.id = marker_id
    m.type = Marker.TRIANGLE_LIST
    m.action = Marker.ADD
    m.scale.x = 1.0
    m.scale.y = 1.0
    m.scale.z = 1.0
    m.color = bgr_to_color_rgba(color_bgr, alpha=alpha)
    m.pose.orientation.w = 1.0

    vertices = mesh['vertices']
    for face in mesh['faces']:
        for vi in face:
            v = vertices[vi]
            m.points.append(_make_point(float(v[0]), float(v[1]), float(v[2])))
    return m


def build_hybrid_surface_marker_array(results, stamp, frame_id, args):
    markers = MarkerArray()
    markers.markers.append(make_delete_all_marker(stamp, frame_id))
    if not getattr(args, 'publish_hybrid_surface_marker', True):
        return markers
    if not results:
        return markers

    marker_id = 0
    for r in results:
        mesh = r.get('surface_mesh')
        if mesh is None:
            continue
        label = r['tag']
        inst = r['instance_id']
        ns = f'hybrid_surface/{label}_{inst}'
        markers.markers.append(make_hybrid_surface_marker(
            marker_id, frame_id, stamp, mesh, r['color'],
            alpha=float(getattr(args, 'hybrid_surface_marker_alpha', 0.55)),
            ns=ns))
        marker_id += 1

    return markers
