# Scripting

The Python host is an optional DLL, `rend_pyhost.dll`, that embeds CPython inside
the viewer. Nothing links against it: the viewer loads it from the executable's
directory only when a scene lists `<Script>` nodes or the `--script` flag is
given, and a scene with no scripts never touches Python at all — the DLL and the
CPython runtime it carries can be entirely absent. CMake fetches CPython itself
(no system Python install required), and the whole feature can be turned off at
configure time with `-DREND_PYTHON=OFF`.

Scripts attach to a running viewer in two ways. A scene file can declare
`<Script path="scripts/foo.py"/>` (see `SCENE_FORMAT.md`), resolved relative to
the scene file like mesh paths; the viewer runs every listed script, in order,
on the interpreter's own thread. The viewer's `--script <path>` flag attaches a
script the same way but from the command line, and can be repeated. Scripts
passed with `--script` run **before** any scene script, so a script meant to
supervise or drive the session must be given this way — if it never returns, a
scene script would otherwise never get the (single, sequential) script thread.
A `--script` path that does not exist is logged as a warning and skipped.

## A minimal script

```python
import rend
import time

handle = rend.load_model("model/Xbot/Xbot.gltf", position=(0, 0, 0))
rend.wait_model_ready(handle, timeout=30.0)

angle = 0.0
while not rend.should_quit():
    angle += 0.01
    rend.set_camera(
        position=(5 * (angle % 6.28), 2, 5),
        target=(0, 1, 0),
    )
    time.sleep(1 / 60)
```

Every loop that runs for more than an instant must poll `rend.should_quit()`
and exit when it turns true; that is how the viewer gets a script's thread back
at shutdown.

## API reference

All functions live in the `rend` module. Commands (anything that changes
renderer state) are sent asynchronously — they are queued and take effect on a
later frame, not before the call returns. Several getters read state the
renderer *mirrors* back to the script host on change; that mirror can lag the
frame loop by up to one frame, so a script that reacts to its own command
should poll the getter rather than assume an instant echo.

### Models

| Function | Description |
|---|---|
| `load_model(path, position=(0,0,0), yaw=0.0, scale=1.0, reflective=False) -> handle` | Queues a load. The model does not exist until a `model_ready` event reports success (see `poll_events`/`wait_model_ready`). Raises `ValueError` if `path` is 512 bytes or longer. `reflective` is the runtime equivalent of the scene-XML `reflective="true"` tag; material state is shared per resource, so the *first* load of a given path decides it for every instance, and a mismatched repeat load warns. |
| `set_transform(handle, position, yaw=0.0, scale=1.0)` | Moves, rotates or rescales an existing model. |
| `unload_model(handle)` | Removes a model. |

### Camera

| Function | Description |
|---|---|
| `set_camera(position, target)` | Places the free-fly camera at `position` looking at `target` (converted internally to yaw/pitch, so the user's mouselook continues naturally from the scripted pose). Cancels an in-progress scene `<Camera flyFrom>` fly-in. To animate, send one call per step (about 60 Hz is fine); the user regains full manual control the instant the script stops sending. |
| `get_camera() -> dict \| None` | The camera's latest broadcast pose: `{position: (x, y, z), yaw, pitch, fov_degrees}` (yaw/pitch in radians; yaw 0 looks down -Z and increases toward +X, positive pitch looks up). Mirrored state — answers instantly from cache, but is `None` until the first frame has broadcast a pose, and can lag the frame loop by up to a frame. A script reacting to its own teleport should poll until the mirror reflects it rather than testing immediately. |

### Lighting

| Function | Description |
|---|---|
| `set_sun(direction, color=(1,1,1), intensity=1.0)` | Sets the directional sun (`direction` points from the light toward the scene). Shadow cascades refit automatically. |
| `set_ambient(color)` | Hemispherical ambient tint (default is roughly `(0.30, 0.32, 0.36)`). |
| `set_point_light(index, position, color=(1,1,1), intensity=1.0, radius=10.0, casts_shadows=False)` | Replaces one of 16 point-light slots; `intensity=0` turns a slot off. Replacing a slot drops its baked shadow cube (`casts_shadows` still gates ray-traced occlusion). Scene-declared point lights pre-fill slots at load. |
| `set_point_light_scale(scale)` | A global multiplier over every point light's authored intensity — the day/night control; `scale=0` turns all of them off. All lighting commands are safe to issue every frame; nothing about lighting is baked into a static recording. |

### Sky and time of day

| Function | Description |
|---|---|
| `set_sky_color(color)` | Flat background/sky color used by the per-frame sky pass. |
| `set_time_of_day(t)` | Enables the sky pass's procedural sky and drives its palette from a day phase `t` in `[0, 1)` (0 = sunrise, 0.25 = noon, 0.5 = sunset). A negative value (the default) disables it in favor of the flat sky color. |

### Animation

Scene-declared animated models only. Clip switching is pure CPU state, so none
of it invalidates a static frame recording.

| Function | Description |
|---|---|
| `set_animation(clip, model='', blend=0.25, loop=None)` | Cross-fades to the named clip over `blend` seconds (`0` snaps instantly). `loop=None` keeps the clip's scene-declared loop mode; `True`/`False` overrides it. |
| `set_animation_loop(loop, model='')` | Sets the loop mode of the clip currently playing. `False` plays once and holds the last pose — the right choice for a one-shot gesture or a two-key pose clip, since looping one visibly judders every cycle. glTF carries no loop flag, so this call and scene XML are the only ways to state that intent; the viewer warns at load about any looping clip whose first and last poses disagree. |
| `set_animation_speed(speed, model='')` | `1.0` is normal speed, `0` freezes the clip in place, and a negative value plays it backwards (time wraps at both ends). |
| `pause_animation(paused=True, model='')` | Holds the clip and any running cross-fade, but the pose is still re-evaluated, so a seek or a clip switch while paused shows immediately. Unlike `speed=0`, it remembers the speed to resume at. |
| `set_animation_time(seconds, model='')` | Seeks the clip, wrapped into `[0, duration)`. |
| `list_animations(model='') -> {model: [clip names]}` | Lists clip names per model. |
| `get_animation(model='') -> str \| None` | The playing clip's name. |
| `get_animation_state(model='') -> dict \| None` | `{model, clip, clips, duration, time, speed, paused, loop}`. |

For all of these, `model` is a scene `<Model name>`; the default empty string
targets every animated model. A clip name the target model does not have is
logged by the viewer and silently ignored — no rejection event — so check with
`get_animation()` if a switch does not seem to have happened. The last three
getters read a mirror seeded by animation-state broadcasts sent **only on
change**: `time` is exact right after a seek or a switch, then goes stale while
a clip free-runs, because there is no live per-frame clock for it. Omitting
`model` from `get_animation`/`get_animation_state` only resolves when the scene
has exactly one animated model.

### Graphics settings

State here is mirrored the same way: the viewer broadcasts the full settings
state at startup and on every change, so these getters answer instantly from
cache with no round trip. The mirror seeds on the first frame after the host
starts, so a script that queries immediately on startup should sleep briefly
first.

| Function | Description |
|---|---|
| `list_settings() -> list[str]` | Names of every settings slot. |
| `get_setting(name) -> str \| None` | The active option token, a number formatted as a string, or the literal `'override'` when another setting supersedes this one. |
| `get_setting_options(name) -> list[str]` | What is selectable right now; empty while the slot is overridden, and empty for numeric slots. |
| `get_override_source(name) -> str \| None` | The name of the slot that is overriding this one, if any (for example `'primary'`). |
| `set_setting(name, value)` | Requests a change, asynchronously like every command. A refusal — unknown slot, unlisted option, or an overridden slot — arrives as a `'setting_rejected'` event `{name, value, reason}` from `poll_events()`, not as an exception. |

Slots as of this writing: `primary` (`raster`/`raytraced`), `shadows`
(`cascaded`/`raytraced`), `reflections` (`probe`/`raytraced`),
`frustum_culling`/`lod_selection`/`occlusion_culling`/`occlusion_boxes`
(on/off), and `fog_density` (a float from 0 to 0.15, only meaningful in scenes
with fog). Setting `primary` to `raytraced` overrides every technique and
culling slot.

### Events and lifecycle

| Function | Description |
|---|---|
| `poll_events() -> list[dict]` | Drains cached events, currently `{type: 'model_ready', handle, ok, millis, error}` and `{type: 'setting_rejected', name, value, reason}`. Events a caller does not consume stay cached, so a `wait_model_ready` call elsewhere in the script does not lose events this one doesn't ask for. |
| `wait_model_ready(handle, timeout=30.0) -> dict \| None` | Blocks the calling thread (releasing the GIL between short spins) until the named model's `model_ready` event arrives or the timeout elapses, returning `None` on timeout. |
| `should_quit() -> bool` | `True` once the viewer has asked scripts to stop. Any loop that can run for more than an instant must poll this and exit when it turns true. |
| `log(message)` | Writes a line to the viewer's log, prefixed `[py]`. |

## Threading and lifecycle

One Python host runs per viewer process; a second embedded interpreter is not
supported. Scripts run one after another on a single dedicated thread, each
with a fresh set of `__main__` globals — a script that raises is logged and the
next one still runs. Long waits inside a script are cheap: `time.sleep()`, or
even a busy spin that holds the GIL, blocks only the script thread, never the
renderer's frame loop.

A script may also spawn its own `threading.Thread`s and call `rend.*` functions
from them concurrently — this is supported, because every binding does its work
while holding the GIL. Worker threads should still poll `should_quit()`
themselves (the interpreter's own stop signal only reaches the main thread),
and the main script should join them before returning.

The viewer stops the Python host before it tears down anything else it owns,
and never unloads the DLL itself once loaded, since CPython does not tolerate
being unloaded from a process and reloaded later.
