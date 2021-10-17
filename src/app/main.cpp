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

namespace {

const char *kHelpString = "ReSTIR-OptiX\n" \
    "-w <width>                -- width of window, optional, default is 1200\n" \
    "-h <height>               -- height of window, optional, default is 900\n" \
    "-l <light-strength-scale> -- scaling factor of light strength in the scene, optional, default is 1.0\n" \
    "-s <gltf-path>            -- path of .gltf, required\n";

bool ParseConfig(RestirAppConfig &config, int argc, char *argv[]) {
    bool valid = true;
    bool has_path = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-w") == 0 && i + 1 < argc) {
            ++i;
            config.width = std::atoi(argv[i]);
            if (config.width <= 0) {
                valid = false;
            }
        } else if (strcmp(argv[i], "-h") == 0 && i + 1 < argc) {
            ++i;
            config.height = std::atoi(argv[i]);
            if (config.height <= 0) {
                valid = false;
            }
        } else if (strcmp(argv[i], "-l") == 0 && i + 1 < argc) {
            ++i;
            config.light_strength_scale = std::atof(argv[i]);
            if (config.light_strength_scale <= 0.0f) {
                valid = false;
            }
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            ++i;
            config.scene_path = argv[i];
            has_path = true;
        }
    }

    return valid && has_path;
}

}

int main(int argc, char *argv[]) {
    try {
        RestirAppConfig config = {};
        if (!ParseConfig(config, argc, argv)) {
            fmt::print(stderr, "Invalid command line arguments\n{}", kHelpString);
            return 0;
        }

        fmt::print("Initializing...\n");
        RestirApp app(config);

        fmt::print("Initialization is finished, begin to run...\n");
        app.MainLoop();
    } catch (const std::runtime_error &err) {
        fmt::print(stderr, "Runtime error: {}\n", err.what());
    }

    return 0;
}