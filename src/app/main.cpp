#include <stdexcept>

#include "fmt/core.h"
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "tiny_gltf.h"
#undef TINYGLTF_IMPLEMENTATION
#undef STB_IMAGE_IMPLEMENTATION
#undef STB_IMAGE_WRITE_IMPLEMENTATION

#include "restir_optix.hpp"
#include "app/app.hpp"

int main(int argc, char *argv[]) {
    try {
        RestirAppConfig config;
        config.width = 1200;
        config.height = 900;
        config.title = "ReSTIR-OptiX";
        config.scene_path = fmt::format("{}/cornellBox/cornellBox.gltf", kProjectAssetsDir);

        fmt::print("Initializing...\n");
        RestirApp app(config);

        fmt::print("Initialization is finished, begin to run...\n");
        app.MainLoop();
    } catch (const std::runtime_error &err) {
        fmt::print(stderr, "Runtime error: {}\n", err.what());
    }

    return 0;
}