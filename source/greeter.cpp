#include <bairstow/greeter.h>
#include <fmt/format.h>  // for format

#include <type_traits>  // for move

using namespace bairstow;
using namespace std;

Bairstow::Bairstow(string _name) : name(move(_name)) {}

auto Bairstow::greet(LanguageCode lang) const -> std::string {
    switch (lang) {
        default:
        case LanguageCode::EN:
            return fmt::format("Hello, {}!", name);
        case LanguageCode::DE:
            return fmt::format("Hallo {}!", name);
        case LanguageCode::ES:
            return fmt::format("¡Hola {}!", name);
        case LanguageCode::FR:
            return fmt::format("Bonjour {}!", name);
    }
}
