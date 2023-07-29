#pragma once

#include <iosfwd> // for string
#include <string> // for basic_string

namespace bairstow {

/**  Language codes to be used with the Bairstow class */
enum class LanguageCode { EN, DE, ES, FR };

/**
 * @brief A class for saying hello in multiple languages
 */
class Bairstow {
    std::string name;

  public:
    /**
     * @brief Creates a new bairstow
     * @param name the name to greet
     */
    Bairstow(std::string name);

    /**
     * @brief Creates a localized string containing the greeting
     * @param lang the language to greet in
     * @return a string containing the greeting
     */
    std::string greet(LanguageCode lang = LanguageCode::EN) const;
};

} // namespace bairstow
