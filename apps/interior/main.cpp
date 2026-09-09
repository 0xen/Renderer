// interior_gen — experimental procedural interior generator (app layer, not
// part of the renderer). Generates a tile-based interior into a scene folder
// the viewer can run directly:
//
//   - a 2D tile grid (tiles are 3x3 m) BSP-partitioned into rooms of
//     1x1 .. 4x4 tiles; doorways come from a spanning tree over the room
//     adjacency graph plus a few extra loops,
//   - basic GLB tile models it authors itself (floor+ceiling tile, wall
//     segment, doorway segment, ground plane) — untextured PBR factors,
//   - interior_layout.json (the placed instances) + interior.xml (camera,
//     lights, static ground, the build script) + scripts/build_interior.py,
//     which spawns every instance through the viewer's runtime path so
//     repeated models share geometry (true instancing: a handful of draw
//     entries for the whole dungeon).
//
// The "3D grid" milestone (stairwell rooms poking into vertical levels) will
// grow out of the same layout structures; tile models are meant to become
// XML-defined assets (footprint in tiles + doorway edges) later — the
// models/instances split in the JSON already reflects that.
//
//   interior_gen [--seed N] [--grid W H] [--out dir]

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr float kTileSize = 3.0f;   // meters per grid tile
constexpr float kRoomHeight = 3.0f; // floor-to-ceiling
constexpr float kWallThickness = 0.2f;
constexpr int kMaxRoomSide = 4; // rooms are 1x1 .. 4x4 tiles
constexpr int kMaxPointLights = 16;

struct Vec3 {
    float x = 0.0f, y = 0.0f, z = 0.0f;
};

// ---------------------------------------------------------------- GLB writer

struct MaterialDef {
    std::string name;
    float r = 1.0f, g = 1.0f, b = 1.0f;
    float roughness = 0.9f;
    float metallic = 0.0f;
};

struct MeshData {
    std::string name;
    int material = 0;
    std::vector<float> pos; // xyz
    std::vector<float> nrm; // xyz
    std::vector<float> uv;  // uv
    std::vector<std::uint32_t> idx;
};

// Quad from a corner and two edge vectors; front face is CCW around
// cross(u, v) (glTF right-handed winding), which must equal n.
void appendQuad(MeshData& m, Vec3 o, Vec3 u, Vec3 v, Vec3 n) {
    const std::uint32_t base = static_cast<std::uint32_t>(m.pos.size() / 3);
    const std::array<Vec3, 4> corners = {
        Vec3{o.x, o.y, o.z},
        Vec3{o.x + u.x, o.y + u.y, o.z + u.z},
        Vec3{o.x + u.x + v.x, o.y + u.y + v.y, o.z + u.z + v.z},
        Vec3{o.x + v.x, o.y + v.y, o.z + v.z},
    };
    const std::array<std::array<float, 2>, 4> uvs = {{{0, 0}, {1, 0}, {1, 1}, {0, 1}}};
    for (std::size_t i = 0; i < 4; ++i) {
        m.pos.insert(m.pos.end(), {corners[i].x, corners[i].y, corners[i].z});
        m.nrm.insert(m.nrm.end(), {n.x, n.y, n.z});
        m.uv.insert(m.uv.end(), {uvs[i][0], uvs[i][1]});
    }
    m.idx.insert(m.idx.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
}

// Axis-aligned box, all six faces outward.
void appendBox(MeshData& m, Vec3 mn, Vec3 mx) {
    const float w = mx.x - mn.x, h = mx.y - mn.y, d = mx.z - mn.z;
    appendQuad(m, {mx.x, mn.y, mx.z}, {0, 0, -d}, {0, h, 0}, {1, 0, 0});  // +X
    appendQuad(m, {mn.x, mn.y, mn.z}, {0, 0, d}, {0, h, 0}, {-1, 0, 0});  // -X
    appendQuad(m, {mn.x, mx.y, mx.z}, {w, 0, 0}, {0, 0, -d}, {0, 1, 0});  // +Y
    appendQuad(m, {mn.x, mn.y, mn.z}, {w, 0, 0}, {0, 0, d}, {0, -1, 0});  // -Y
    appendQuad(m, {mn.x, mn.y, mx.z}, {w, 0, 0}, {0, h, 0}, {0, 0, 1});   // +Z
    appendQuad(m, {mx.x, mn.y, mn.z}, {-w, 0, 0}, {0, h, 0}, {0, 0, -1}); // -Z
}

bool writeGlb(const std::filesystem::path& file, const std::vector<MeshData>& meshes,
              const std::vector<MaterialDef>& materials) {
    // Binary chunk: per mesh, one view each for indices / positions /
    // normals / uvs. Everything is 4-byte aligned by construction.
    std::vector<std::uint8_t> bin;
    struct View {
        std::size_t offset = 0, length = 0;
    };
    std::vector<View> views;
    auto addView = [&](const void* data, std::size_t bytes) {
        const std::size_t offset = bin.size();
        bin.resize(offset + bytes);
        std::memcpy(bin.data() + offset, data, bytes);
        views.push_back({offset, bytes});
        return views.size() - 1;
    };

    std::string accessors, bufferViews, meshJson, nodeJson, materialJson;
    std::size_t accessorCount = 0;
    for (const MeshData& m : meshes) {
        const std::size_t vi = addView(m.idx.data(), m.idx.size() * 4);
        const std::size_t vp = addView(m.pos.data(), m.pos.size() * 4);
        const std::size_t vn = addView(m.nrm.data(), m.nrm.size() * 4);
        const std::size_t vt = addView(m.uv.data(), m.uv.size() * 4);
        Vec3 mn{1e9f, 1e9f, 1e9f}, mx{-1e9f, -1e9f, -1e9f};
        for (std::size_t i = 0; i + 2 < m.pos.size(); i += 3) {
            mn.x = std::min(mn.x, m.pos[i]);
            mn.y = std::min(mn.y, m.pos[i + 1]);
            mn.z = std::min(mn.z, m.pos[i + 2]);
            mx.x = std::max(mx.x, m.pos[i]);
            mx.y = std::max(mx.y, m.pos[i + 1]);
            mx.z = std::max(mx.z, m.pos[i + 2]);
        }
        const std::size_t vertexCount = m.pos.size() / 3;
        const std::size_t ai = accessorCount++; // indices
        const std::size_t ap = accessorCount++; // positions
        const std::size_t an = accessorCount++; // normals
        const std::size_t at = accessorCount++; // uvs
        auto sep = [](std::string& s) {
            if (!s.empty()) s += ',';
        };
        sep(accessors);
        accessors += std::format(
            R"({{"bufferView":{},"componentType":5125,"count":{},"type":"SCALAR"}},)"
            R"({{"bufferView":{},"componentType":5126,"count":{},"type":"VEC3","min":[{:.4f},{:.4f},{:.4f}],"max":[{:.4f},{:.4f},{:.4f}]}},)"
            R"({{"bufferView":{},"componentType":5126,"count":{},"type":"VEC3"}},)"
            R"({{"bufferView":{},"componentType":5126,"count":{},"type":"VEC2"}})",
            vi, m.idx.size(), vp, vertexCount, mn.x, mn.y, mn.z, mx.x, mx.y, mx.z, vn,
            vertexCount, vt, vertexCount);
        sep(meshJson);
        meshJson += std::format(
            R"({{"name":"{}","primitives":[{{"attributes":{{"POSITION":{},"NORMAL":{},"TEXCOORD_0":{}}},"indices":{},"material":{}}}]}})",
            m.name, ap, an, at, ai, m.material);
        sep(nodeJson);
        nodeJson += std::format(R"({{"name":"{}","mesh":{}}})", m.name, &m - meshes.data());
    }
    for (const View& v : views) {
        if (!bufferViews.empty()) bufferViews += ',';
        bufferViews +=
            std::format(R"({{"buffer":0,"byteOffset":{},"byteLength":{}}})", v.offset, v.length);
    }
    for (const MaterialDef& mat : materials) {
        if (!materialJson.empty()) materialJson += ',';
        materialJson += std::format(
            R"({{"name":"{}","pbrMetallicRoughness":{{"baseColorFactor":[{:.3f},{:.3f},{:.3f},1.0],"metallicFactor":{:.2f},"roughnessFactor":{:.2f}}}}})",
            mat.name, mat.r, mat.g, mat.b, mat.metallic, mat.roughness);
    }
    std::string sceneNodes;
    for (std::size_t i = 0; i < meshes.size(); ++i) {
        if (!sceneNodes.empty()) sceneNodes += ',';
        sceneNodes += std::to_string(i);
    }
    std::string json = std::format(
        R"({{"asset":{{"version":"2.0","generator":"interior_gen"}},"scene":0,"scenes":[{{"nodes":[{}]}}],"nodes":[{}],"meshes":[{}],"materials":[{}],"accessors":[{}],"bufferViews":[{}],"buffers":[{{"byteLength":{}}}]}})",
        sceneNodes, nodeJson, meshJson, materialJson, accessors, bufferViews, bin.size());

    while (json.size() % 4 != 0) json += ' ';
    while (bin.size() % 4 != 0) bin.push_back(0);

    std::ofstream out(file, std::ios::binary);
    if (!out) return false;
    auto u32 = [&](std::uint32_t v) { out.write(reinterpret_cast<const char*>(&v), 4); };
    u32(0x46546C67); // 'glTF'
    u32(2);
    u32(static_cast<std::uint32_t>(12 + 8 + json.size() + 8 + bin.size()));
    u32(static_cast<std::uint32_t>(json.size()));
    u32(0x4E4F534A); // 'JSON'
    out.write(json.data(), static_cast<std::streamsize>(json.size()));
    u32(static_cast<std::uint32_t>(bin.size()));
    u32(0x004E4942); // 'BIN'
    out.write(reinterpret_cast<const char*>(bin.data()), static_cast<std::streamsize>(bin.size()));
    return out.good();
}

// ---------------------------------------------------------------- tile models

bool writeTileModels(const std::filesystem::path& modelDir, float groundExtent) {
    const float half = kTileSize * 0.5f;
    const float t = kWallThickness * 0.5f;

    // tile.glb: floor + ceiling of one 3x3 m cell (yaw-symmetric).
    MeshData floor{"floor", 0};
    appendQuad(floor, {-half, 0, half}, {kTileSize, 0, 0}, {0, 0, -kTileSize}, {0, 1, 0});
    MeshData ceiling{"ceiling", 1};
    appendQuad(ceiling, {-half, kRoomHeight, -half}, {kTileSize, 0, 0}, {0, 0, kTileSize},
               {0, -1, 0});
    const bool tileOk = writeGlb(modelDir / "tile.glb", {floor, ceiling},
                                 {{"floor", 0.40f, 0.37f, 0.34f, 0.92f},
                                  {"ceiling", 0.80f, 0.79f, 0.76f, 0.96f}});

    // wall.glb: one 3 m edge segment along X, straddling the tile boundary.
    MeshData wall{"wall", 0};
    appendBox(wall, {-half, 0, -t}, {half, kRoomHeight, t});
    const bool wallOk =
        writeGlb(modelDir / "wall.glb", {wall}, {{"plaster", 0.72f, 0.69f, 0.64f, 0.88f}});

    // door.glb: wall segment with a 1.2 x 2.2 m opening (two jambs + lintel).
    MeshData door{"doorway", 0};
    appendBox(door, {-half, 0, -t}, {-0.6f, kRoomHeight, t});
    appendBox(door, {0.6f, 0, -t}, {half, kRoomHeight, t});
    appendBox(door, {-0.6f, 2.2f, -t}, {0.6f, kRoomHeight, t});
    const bool doorOk =
        writeGlb(modelDir / "door.glb", {door}, {{"plaster", 0.66f, 0.60f, 0.54f, 0.85f}});

    // ground.glb: static plane under the whole building.
    MeshData ground{"ground", 0};
    appendQuad(ground, {-groundExtent, 0, groundExtent}, {2 * groundExtent, 0, 0},
               {0, 0, -2 * groundExtent}, {0, 1, 0});
    const bool groundOk =
        writeGlb(modelDir / "ground.glb", {ground}, {{"ground", 0.28f, 0.32f, 0.26f, 0.95f}});

    return tileOk && wallOk && doorOk && groundOk;
}

// ---------------------------------------------------------------- layout

struct Room {
    int x = 0, z = 0, w = 1, h = 1; // in tiles
};

struct Instance {
    std::string model;
    float x = 0, y = 0, z = 0;
    float yaw = 0;
};

struct PointLight {
    float x = 0, y = 0, z = 0;
    float radius = 8.0f;
};

struct Layout {
    int gridW = 0, gridH = 0;
    std::vector<Room> rooms;
    std::vector<Instance> instances;
    std::vector<PointLight> lights;
    std::size_t doorCount = 0;
    Vec3 cameraPos{}, cameraTarget{};
};

void splitRegion(std::mt19937& rng, int x, int z, int w, int h, std::vector<Room>& rooms) {
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    const bool mustSplit = w > kMaxRoomSide || h > kMaxRoomSide;
    if (!mustSplit) {
        const bool splittable = w >= 2 || h >= 2;
        if (!splittable || uni(rng) < 0.55f) {
            rooms.push_back({x, z, w, h});
            return;
        }
    }
    bool splitX;
    if ((w > kMaxRoomSide) != (h > kMaxRoomSide)) splitX = w > kMaxRoomSide;
    else if (w != h) splitX = w > h;
    else splitX = uni(rng) < 0.5f;
    if (splitX && w < 2) splitX = false;
    if (!splitX && h < 2) splitX = true;
    if (splitX) {
        const int cut = std::uniform_int_distribution<int>(1, w - 1)(rng);
        splitRegion(rng, x, z, cut, h, rooms);
        splitRegion(rng, x + cut, z, w - cut, h, rooms);
    } else {
        const int cut = std::uniform_int_distribution<int>(1, h - 1)(rng);
        splitRegion(rng, x, z, w, cut, rooms);
        splitRegion(rng, x, z + cut, w, h - cut, rooms);
    }
}

struct UnionFind {
    std::vector<int> parent;
    explicit UnionFind(std::size_t n) : parent(n) { std::iota(parent.begin(), parent.end(), 0); }
    int find(int a) {
        while (parent[static_cast<std::size_t>(a)] != a) a = parent[static_cast<std::size_t>(a)];
        return a;
    }
    bool unite(int a, int b) {
        a = find(a);
        b = find(b);
        if (a == b) return false;
        parent[static_cast<std::size_t>(a)] = b;
        return true;
    }
};

Layout generateLayout(int gridW, int gridH, std::uint32_t seed) {
    Layout out;
    out.gridW = gridW;
    out.gridH = gridH;
    std::mt19937 rng(seed);
    splitRegion(rng, 0, 0, gridW, gridH, out.rooms);

    std::vector<int> roomOf(static_cast<std::size_t>(gridW * gridH), -1);
    for (std::size_t r = 0; r < out.rooms.size(); ++r) {
        const Room& room = out.rooms[r];
        for (int z = room.z; z < room.z + room.h; ++z)
            for (int x = room.x; x < room.x + room.w; ++x)
                roomOf[static_cast<std::size_t>(z * gridW + x)] = static_cast<int>(r);
    }

    // World mapping: building centered on the origin, tile (cx, cz) center
    // at (wx0 + (cx+.5)*3, 0, wz0 + (cz+.5)*3).
    const float wx0 = -static_cast<float>(gridW) * kTileSize * 0.5f;
    const float wz0 = -static_cast<float>(gridH) * kTileSize * 0.5f;
    auto cellCenterX = [&](int cx) { return wx0 + (static_cast<float>(cx) + 0.5f) * kTileSize; };
    auto cellCenterZ = [&](int cz) { return wz0 + (static_cast<float>(cz) + 0.5f) * kTileSize; };

    // Floor/ceiling tile per cell.
    for (int cz = 0; cz < gridH; ++cz)
        for (int cx = 0; cx < gridW; ++cx)
            out.instances.push_back({"tile", cellCenterX(cx), 0, cellCenterZ(cz), 0});

    // Interior edges between different rooms are doorway candidates; the
    // rest (plus the outer boundary) become walls. An edge along Z gets the
    // X-authored wall model at yaw 90.
    struct EdgeSpot {
        float x, z, yaw;
    };
    std::map<std::pair<int, int>, std::vector<EdgeSpot>> candidates;
    std::vector<EdgeSpot> walls;
    for (int cz = 0; cz < gridH; ++cz) {
        for (int cx = 0; cx < gridW; ++cx) {
            const int self = roomOf[static_cast<std::size_t>(cz * gridW + cx)];
            // outer boundary
            if (cx == 0) walls.push_back({wx0, cellCenterZ(cz), 90});
            if (cz == 0) walls.push_back({cellCenterX(cx), wz0, 0});
            // east edge
            {
                const EdgeSpot spot{wx0 + static_cast<float>(cx + 1) * kTileSize, cellCenterZ(cz),
                                    90};
                if (cx + 1 == gridW) walls.push_back(spot);
                else {
                    const int other = roomOf[static_cast<std::size_t>(cz * gridW + cx + 1)];
                    if (other != self)
                        candidates[{std::min(self, other), std::max(self, other)}].push_back(spot);
                }
            }
            // south edge (+Z)
            {
                const EdgeSpot spot{cellCenterX(cx), wz0 + static_cast<float>(cz + 1) * kTileSize,
                                    0};
                if (cz + 1 == gridH) walls.push_back(spot);
                else {
                    const int other = roomOf[static_cast<std::size_t>((cz + 1) * gridW + cx)];
                    if (other != self)
                        candidates[{std::min(self, other), std::max(self, other)}].push_back(spot);
                }
            }
        }
    }

    // Doors: spanning tree over the adjacency graph (every room reachable),
    // then extra loop doors on a fraction of the remaining adjacencies.
    std::vector<std::pair<int, int>> pairs;
    pairs.reserve(candidates.size());
    for (const auto& [key, spots] : candidates) pairs.push_back(key);
    std::shuffle(pairs.begin(), pairs.end(), rng);
    UnionFind uf(out.rooms.size());
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    std::vector<EdgeSpot> doors;
    std::map<int, EdgeSpot> firstDoorOfRoom;
    for (const auto& key : pairs) {
        auto& spots = candidates[key];
        const bool tree = uf.unite(key.first, key.second);
        const bool loop = !tree && uni(rng) < 0.2f;
        std::size_t doorIndex = spots.size(); // = none
        if (tree || loop) {
            doorIndex = std::uniform_int_distribution<std::size_t>(0, spots.size() - 1)(rng);
            doors.push_back(spots[doorIndex]);
            firstDoorOfRoom.try_emplace(key.first, spots[doorIndex]);
            firstDoorOfRoom.try_emplace(key.second, spots[doorIndex]);
        }
        for (std::size_t i = 0; i < spots.size(); ++i)
            if (i != doorIndex) walls.push_back(spots[i]);
    }
    out.doorCount = doors.size();
    for (const EdgeSpot& w : walls) out.instances.push_back({"wall", w.x, 0, w.z, w.yaw});
    for (const EdgeSpot& d : doors) out.instances.push_back({"door", d.x, 0, d.z, d.yaw});

    // Point lights in the largest rooms (renderer supports 16 slots).
    std::vector<std::size_t> byArea(out.rooms.size());
    std::iota(byArea.begin(), byArea.end(), 0);
    std::sort(byArea.begin(), byArea.end(), [&](std::size_t a, std::size_t b) {
        return out.rooms[a].w * out.rooms[a].h > out.rooms[b].w * out.rooms[b].h;
    });
    for (std::size_t i = 0; i < byArea.size() && i < kMaxPointLights; ++i) {
        const Room& room = out.rooms[byArea[i]];
        const float lx = wx0 + (static_cast<float>(room.x) + static_cast<float>(room.w) * 0.5f) *
                                   kTileSize;
        const float lz = wz0 + (static_cast<float>(room.z) + static_cast<float>(room.h) * 0.5f) *
                                   kTileSize;
        const float reach = static_cast<float>(std::max(room.w, room.h)) * kTileSize * 1.2f + 2.0f;
        out.lights.push_back({lx, kRoomHeight - 0.4f, lz, reach});
    }

    // Camera: eye height in the largest room, aimed at one of its doors.
    const Room& camRoom = out.rooms[byArea.front()];
    out.cameraPos = {wx0 + (static_cast<float>(camRoom.x) + static_cast<float>(camRoom.w) * 0.5f) *
                              kTileSize,
                     1.7f,
                     wz0 + (static_cast<float>(camRoom.z) + static_cast<float>(camRoom.h) * 0.5f) *
                              kTileSize};
    const auto door = firstDoorOfRoom.find(static_cast<int>(byArea.front()));
    if (door != firstDoorOfRoom.end())
        out.cameraTarget = {door->second.x, 1.4f, door->second.z};
    else
        out.cameraTarget = {out.cameraPos.x + 1.0f, 1.4f, out.cameraPos.z};
    return out;
}

// ---------------------------------------------------------------- scene files

bool writeLayoutJson(const std::filesystem::path& file, const Layout& layout) {
    std::string inst;
    for (const Instance& i : layout.instances) {
        if (!inst.empty()) inst += ",\n    ";
        inst += std::format(
            R"({{"model":"{}","position":[{:.3f},{:.3f},{:.3f}],"yaw":{:.1f}}})", i.model, i.x,
            i.y, i.z, i.yaw);
    }
    std::ofstream out(file, std::ios::binary);
    out << std::format(
        "{{\n  \"models\": {{\n    \"tile\": \"model/tile.glb\",\n    \"wall\": "
        "\"model/wall.glb\",\n    \"door\": \"model/door.glb\"\n  }},\n  \"roomCount\": {},\n  "
        "\"doorCount\": {},\n  \"instances\": [\n    {}\n  ]\n}}\n",
        layout.rooms.size(), layout.doorCount, inst);
    return out.good();
}

bool writeSceneXml(const std::filesystem::path& file, const Layout& layout) {
    std::string lights;
    for (const PointLight& l : layout.lights) {
        lights += std::format(
            "    <Light type=\"point\" position=\"{:.2f} {:.2f} {:.2f}\" radius=\"{:.1f}\" "
            "color=\"1.0 0.85 0.62\" intensity=\"2.6\" castsShadows=\"false\" />\n",
            l.x, l.y, l.z, l.radius);
    }
    std::ofstream out(file, std::ios::binary);
    out << std::format(
        "<Scene name=\"interior\">\n"
        "    <Camera position=\"{:.2f} {:.2f} {:.2f}\" target=\"{:.2f} {:.2f} {:.2f}\" "
        "fovDegrees=\"70\" />\n"
        "    <Light type=\"directional\" direction=\"0.4 -1.0 0.3\" color=\"1.0 0.96 0.9\" "
        "intensity=\"0.8\" castsShadows=\"true\" />\n"
        "{}"
        "    <Model name=\"ground\">\n"
        "        <Mesh path=\"model/ground.glb\" />\n"
        "        <Transform position=\"0 -0.05 0\" />\n"
        "    </Model>\n"
        "    <Script path=\"scripts/build_interior.py\" />\n"
        "</Scene>\n",
        layout.cameraPos.x, layout.cameraPos.y, layout.cameraPos.z, layout.cameraTarget.x,
        layout.cameraTarget.y, layout.cameraTarget.z, lights);
    return out.good();
}

bool writeBuildScript(const std::filesystem::path& file) {
    const char* script = R"PY(# Spawns the generated interior layout through the runtime instancing
# path: repeats of a model path share geometry, so the whole dungeon costs
# a handful of draw entries. Generated by interior_gen; edits are lost on
# regeneration.
import json
import os

import rend

sceneDir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
with open(os.path.join(sceneDir, "interior_layout.json"), "r", encoding="utf-8") as f:
    layout = json.load(f)

models = layout["models"]
firstHandle = {}
lastHandle = None
count = 0
for inst in layout["instances"]:
    if rend.should_quit():
        break
    name = inst["model"]
    path = os.path.join(sceneDir, models[name])
    handle = rend.load_model(path, tuple(inst["position"]), inst.get("yaw", 0.0))
    lastHandle = handle
    count += 1
    if name not in firstHandle:
        # Let the first instance of each model finish importing before the
        # repeats queue up behind it.
        firstHandle[name] = handle
        result = rend.wait_model_ready(handle, 30.0)
        if not result or not result["ok"]:
            rend.log("interior: FAILED to load " + path)

if lastHandle is not None:
    rend.wait_model_ready(lastHandle, 60.0)
rend.log(
    "interior: spawned %d instances (%d rooms, %d doors, %d unique models)"
    % (count, layout["roomCount"], layout["doorCount"], len(models))
)
)PY";
    std::ofstream out(file, std::ios::binary);
    out << script;
    return out.good();
}

bool writeLauncher(const std::filesystem::path& file) {
    std::ofstream out(file, std::ios::binary); // binary => explicit CRLF
    out << "@echo off\r\n"
        << "C:\\github\\Renderer\\build\\bin\\Debug\\viewer.exe --static \"%~dp0interior.xml\"\r\n";
    return out.good();
}

bool writeReadme(const std::filesystem::path& file) {
    std::ofstream out(file, std::ios::binary);
    out << "All models in model/ are generated by interior_gen (apps/interior in the\n"
           "Renderer repo). No external assets; regenerate with interior_gen.\n";
    return out.good();
}

} // namespace

int main(int argc, char** argv) {
    std::uint32_t seed = 1;
    int gridW = 10, gridH = 10;
    std::filesystem::path outDir = "C:/github/scenes/interior";
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--seed" && i + 1 < argc) seed = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        else if (arg == "--grid" && i + 2 < argc) {
            gridW = std::stoi(argv[++i]);
            gridH = std::stoi(argv[++i]);
        } else if (arg == "--out" && i + 1 < argc) outDir = argv[++i];
        else {
            std::cout << "usage: interior_gen [--seed N] [--grid W H] [--out dir]\n";
            return arg == "--help" ? 0 : 1;
        }
    }
    gridW = std::clamp(gridW, 1, 32);
    gridH = std::clamp(gridH, 1, 32);

    std::error_code ec;
    std::filesystem::create_directories(outDir / "model", ec);
    std::filesystem::create_directories(outDir / "scripts", ec);
    if (ec) {
        std::cout << "failed to create " << outDir << ": " << ec.message() << "\n";
        return 1;
    }

    const Layout layout = generateLayout(gridW, gridH, seed);
    const float groundExtent =
        static_cast<float>(std::max(gridW, gridH)) * kTileSize * 0.5f + 12.0f;

    bool ok = writeTileModels(outDir / "model", groundExtent);
    ok = writeLayoutJson(outDir / "interior_layout.json", layout) && ok;
    ok = writeSceneXml(outDir / "interior.xml", layout) && ok;
    ok = writeBuildScript(outDir / "scripts" / "build_interior.py") && ok;
    ok = writeLauncher(outDir / "run_interior.bat") && ok;
    ok = writeReadme(outDir / "MODEL_README.md") && ok;
    if (!ok) {
        std::cout << "interior_gen: one or more outputs failed to write\n";
        return 1;
    }

    std::size_t wallCount = 0, doorCount = 0;
    for (const Instance& i : layout.instances) {
        wallCount += i.model == "wall";
        doorCount += i.model == "door";
    }
    std::cout << std::format(
        "interior_gen: seed {} grid {}x{} -> {} rooms, {} doors, {} walls, {} tiles, {} point "
        "lights\n  {} instances total -> {}\n",
        seed, gridW, gridH, layout.rooms.size(), layout.doorCount, wallCount,
        static_cast<std::size_t>(gridW) * static_cast<std::size_t>(gridH), layout.lights.size(),
        layout.instances.size(), outDir.string());
    return 0;
}
