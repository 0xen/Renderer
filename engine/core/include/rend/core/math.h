#pragma once

#include <array>
#include <cmath>

// Minimal fixed-function-free math: just what the renderer needs. Matrices
// are column-major float[16] (m[column * 4 + row]) so they memcpy straight
// into HLSL's default column_major layout.
namespace rend::math {

struct Vec3 {
    float x = 0.0f, y = 0.0f, z = 0.0f;
};

using Mat4 = std::array<float, 16>;

inline Vec3 sub(const Vec3& a, const Vec3& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline float dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline Vec3 normalize(const Vec3& v) {
    const float len = std::sqrt(dot(v, v));
    return len > 0.0f ? Vec3{v.x / len, v.y / len, v.z / len} : v;
}

inline Mat4 mul(const Mat4& a, const Mat4& b) {
    Mat4 out{};
    for (int c = 0; c < 4; ++c) {
        for (int r = 0; r < 4; ++r) {
            float sum = 0.0f;
            for (int k = 0; k < 4; ++k) {
                sum += a[k * 4 + r] * b[c * 4 + k];
            }
            out[c * 4 + r] = sum;
        }
    }
    return out;
}

// Right-handed view matrix, camera looking down -Z.
inline Mat4 lookAt(const Vec3& eye, const Vec3& target, const Vec3& up) {
    const Vec3 f = normalize(sub(target, eye));
    const Vec3 s = normalize(cross(f, up));
    const Vec3 u = cross(s, f);
    Mat4 m{};
    m[0] = s.x;
    m[4] = s.y;
    m[8] = s.z;
    m[12] = -dot(s, eye);
    m[1] = u.x;
    m[5] = u.y;
    m[9] = u.z;
    m[13] = -dot(u, eye);
    m[2] = -f.x;
    m[6] = -f.y;
    m[10] = -f.z;
    m[14] = dot(f, eye);
    m[15] = 1.0f;
    return m;
}

// Vulkan-convention projection: depth 0..1, clip-space +Y down (the Y flip
// is baked in, so no negative-viewport tricks; note it reverses winding).
inline Mat4 perspective(float fovYRadians, float aspect, float nearPlane, float farPlane) {
    const float t = std::tan(fovYRadians * 0.5f);
    Mat4 m{};
    m[0] = 1.0f / (aspect * t);
    m[5] = -1.0f / t;
    m[10] = farPlane / (nearPlane - farPlane);
    m[11] = -1.0f;
    m[14] = (farPlane * nearPlane) / (nearPlane - farPlane);
    return m;
}

// Vulkan-convention orthographic: depth 0..1, +Y-down flip baked in like
// perspective(). Near/far follow the right-handed camera (-Z forward).
inline Mat4 orthographic(float left, float right, float bottom, float top, float nearPlane,
                         float farPlane) {
    Mat4 m{};
    m[0] = 2.0f / (right - left);
    m[5] = -2.0f / (top - bottom);
    m[10] = 1.0f / (nearPlane - farPlane);
    m[12] = -(right + left) / (right - left);
    m[13] = (top + bottom) / (top - bottom);
    m[14] = nearPlane / (nearPlane - farPlane);
    m[15] = 1.0f;
    return m;
}

} // namespace rend::math
