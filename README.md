# ReSTIR-OptiX

Reproduce [Spatiotemporal Reservoir Resampling for Real-Time Ray Tracing with Dynamic Direct Lighting](https://research.nvidia.com/publication/2020-07_Spatiotemporal-reservoir-resampling) in OptiX.

This program will load a gltf scene (path is currently hard coded) and render it using ReSTIR. Emissive meshes will be treated as light sources. A simple ImGui window can be used to modify some parameters of ReSTIR.

Environment map is currently not supported.

Unbiased version is now implemented with some bus when using with visibility reuse together;

## Dependencies

One may download and install these through site of NVIDIA.

* CUDA
* OptiX 7 SDK

These dependencies are added as git submodules.

* [tinygltf](https://github.com/syoyo/tinygltf)
* [mikktspace](https://github.com/mmikk/MikkTSpace)
* [pep-cuda-math](https://github.com/PepcyCh/pep-cuda-math)

One may install these dependencies manually or use [vcpkg](https://github.com/microsoft/vcpkg).

* [fmt](https://github.com/fmtlib/fmt)
* [glfw3](https://github.com/glfw/glfw)
* [ImGui](https://github.com/ocornut/imgui)

Besides, C++20 and OpenGL 4.5 are used.

## ReSTIR

4 Passes

1. G-buffer
    * in `src/shaders/gbuffer.vert` & `src/shaders/gbuffer.frag`
2. Sample to lights, visibility reuse, temporal reuse
    * in `src/raytracing/sample.cu`
3. Spatial reuse
    * in `src/raytracing/spatial.cu`
4. Lighting
    * in `src/raytracing/lighting.cu`