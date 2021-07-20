#include <doctest/doctest.h>
#include <lmi/greeter.h>
#include <lmi/version.h>

#include <string>

TEST_CASE("Lmi") {
    using namespace lmi;

    Lmi lmi("Tests");

    CHECK(lmi.greet(LanguageCode::EN) == "Hello, Tests!");
    CHECK(lmi.greet(LanguageCode::DE) == "Hallo Tests!");
    CHECK(lmi.greet(LanguageCode::ES) == "¡Hola Tests!");
    CHECK(lmi.greet(LanguageCode::FR) == "Bonjour Tests!");
}

TEST_CASE("Lmi version") {
    static_assert(std::string_view(LMI_VERSION) == std::string_view("1.0"));
    CHECK(std::string(LMI_VERSION) == std::string("1.0"));
}
