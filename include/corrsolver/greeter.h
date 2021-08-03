#pragma once

#include <string>

namespace corrsolver {

    /**  Language codes to be used with the CorrSolver class */
    enum class LanguageCode { EN, DE, ES, FR };

    /**
     * @brief A class for saying hello in multiple languages
     */
    class CorrSolver {
        std::string name;

      public:
        /**
         * @brief Creates a new corrsolver
         * @param name the name to greet
         */
        CorrSolver(std::string name);

        /**
         * @brief Creates a localized string containing the greeting
         * @param lang the language to greet in
         * @return a string containing the greeting
         */
        std::string greet(LanguageCode lang = LanguageCode::EN) const;
    };

}  // namespace corrsolver
