/**
 * 掌心朝屏幕外（掌面正对操作者）的手指列布局。
 *
 * 左右手（掌心朝外）：
 *   左手：拇指在屏幕左侧，小指在右侧
 *   右手：拇指在屏幕右侧，小指在左侧（镜像）
 *
 * 指内自上而下：指尖 →（指中）→ 根弯 →（旋转）→ 侧摆
 *
 * L25：指尖与指中耦合；前端将 tip 标为只读并随 mid。
 */
(function (global) {
  // 掌心朝屏幕外
  const FINGER_ORDER_LEFT = ["thumb", "index", "middle", "ring", "pinky"];
  const FINGER_ORDER_RIGHT = ["pinky", "ring", "middle", "index", "thumb"];

  const LABELS = {
    thumb_base: "根弯",
    thumb_abd: "侧摆",
    thumb_rot: "旋转",
    thumb_mid: "指中",
    thumb_mcp: "指中", // URDF 名 = L25/G20 拇指指中（非根弯）
    thumb_tip: "指尖",
    index_base: "根弯",
    index_abd: "侧摆",
    index_mid: "指中",
    index_tip: "指尖",
    middle_base: "根弯",
    middle_abd: "侧摆",
    middle_mid: "指中",
    middle_tip: "指尖",
    ring_base: "根弯",
    ring_abd: "侧摆",
    ring_mid: "指中",
    ring_tip: "指尖",
    pinky_base: "根弯",
    pinky_abd: "侧摆",
    pinky_mid: "指中",
    pinky_tip: "指尖",
    thumb_cmc_pitch: "拇弯",
    thumb_cmc_yaw: "拇摆",
    index_mcp_pitch: "食弯",
    middle_mcp_pitch: "中弯",
    ring_mcp_pitch: "无弯",
    pinky_mcp_pitch: "小弯",
  };

  const FINGER_TITLE = {
    thumb: "拇指",
    index: "食指",
    middle: "中指",
    ring: "无名",
    pinky: "小指",
    other: "其它",
  };

  function classifyJoint(name) {
    const n = String(name || "").toLowerCase();
    if (n.startsWith("rsv_") || n.includes("rsv")) {
      return { finger: "other", role: "rsv", short: "保留", reserved: true };
    }
    let finger = "other";
    if (n.includes("thumb")) finger = "thumb";
    else if (n.includes("index")) finger = "index";
    else if (n.includes("middle")) finger = "middle";
    else if (n.includes("ring") && !n.includes("middle")) finger = "ring";
    else if (n.includes("pinky") || n.includes("little")) finger = "pinky";

    // 先匹配 tip，避免被其它规则误伤。
    // 注意：不可用 includes("mid")——会误匹配 middle_*（含 "mid" 子串）。
    // thumb_mcp 是 URDF 拇指指中（≠ *_mcp_pitch 根弯）。
    let role = "joint";
    if (/(^|_)(tip|distal)($|_)/.test(n) || /_tip$/.test(n)) role = "tip";
    else if (n === "thumb_mcp" || /(^|_)thumb_mcp($|_)/.test(n)) role = "mid";
    else if (/(^|_)(mid|pip)($|_)/.test(n)) role = "mid";
    else if (/(^|_)(abd|yaw|roll)($|_)/.test(n) || n.includes("yaw") || n.includes("roll"))
      role = "abd";
    else if (/(^|_)rot($|_)/.test(n) || n.endsWith("_rot") || n.includes("rot")) role = "rot";
    else if (
      /(^|_)(base|pitch|mcp)($|_)/.test(n) ||
      n.includes("pitch") ||
      n.includes("mcp") ||
      n.endsWith("_base")
    )
      role = "base";

    return {
      finger,
      role,
      short: LABELS[n] || name,
      reserved: false,
    };
  }

  // 数值越小越靠上：指尖最上，侧摆最底
  const ROLE_ORDER = { tip: 0, mid: 1, base: 2, rot: 3, abd: 4, joint: 5, rsv: 9 };

  function buildHandLayout(jointNames, handSide, handType, coupledJoints) {
    const side = (handSide || "left").toLowerCase() === "right" ? "right" : "left";
    const type = (handType || "o20").toLowerCase();
    const order = side === "right" ? FINGER_ORDER_RIGHT : FINGER_ORDER_LEFT;
    const coupledByIndex = {};
    (coupledJoints || []).forEach((c) => {
      if (c && typeof c.index === "number") coupledByIndex[c.index] = c;
    });

    const buckets = {
      thumb: [],
      index: [],
      middle: [],
      ring: [],
      pinky: [],
      other: [],
    };

    jointNames.forEach((name, index) => {
      const meta = classifyJoint(name);
      const coupled = coupledByIndex[index] || null;
      buckets[meta.finger].push({
        index,
        name,
        ...meta,
        coupled: Boolean(coupled),
        coupledTo: coupled ? coupled.source_index : null,
        coupledLabel: coupled ? coupled.label || "耦合/只读" : "",
      });
    });

    Object.keys(buckets).forEach((k) => {
      buckets[k].sort(
        (a, b) => (ROLE_ORDER[a.role] ?? 5) - (ROLE_ORDER[b.role] ?? 5) || a.index - b.index
      );
    });

    const columns = order.map((finger) => ({
      finger,
      title: FINGER_TITLE[finger],
      joints: buckets[finger],
    }));

    const compact = type === "o6" || type === "l6";
    let hint =
      side === "left"
        ? "掌心朝屏幕外 · 左手：左=拇指 … 右=小指 · 上=指尖 → 根弯 → 下=侧摆"
        : "掌心朝屏幕外 · 右手：左=小指 … 右=拇指 · 上=指尖 → 根弯 → 下=侧摆";
    if (type === "l25" || (coupledJoints && coupledJoints.length)) {
      hint += " · 灰底指尖=耦合只读（随指中）";
    }

    return {
      side,
      type,
      compact,
      columns,
      other: buckets.other,
      hint,
    };
  }

  global.HandLayout = { buildHandLayout, classifyJoint, FINGER_TITLE, ROLE_ORDER };
})(window);
