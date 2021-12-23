#include <bairstow/greeter.h>
#include <fmt/format.h>

using namespace bairstow;
using namespace std;

Bairstow::Bairstow(string _name) : name(move(_name)) {}

std::string Bairstow::greet(LanguageCode lang) const {
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
