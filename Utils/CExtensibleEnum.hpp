#ifndef CEXTENSIBLEENUM_HPP
#define CEXTENSIBLEENUM_HPP

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <stdexcept>


namespace Ikomia
{

    // Primary template — intentionally undefined, forces specialization
    template<typename E>
    struct EnumTraits;


    template<typename E>
    class CExtensibleEnum
    {
        static_assert(std::is_enum_v<E>, "E must be an enum type");
        static_assert(std::is_same_v<std::underlying_type_t<E>, int>, "E must be : int");

        public:

            // Default constructor — required for use in std::vector (e.g. resize)
            CExtensibleEnum() : m_id(static_cast<int>(E{})) {}

            // Implicit construction from the base enum — retro-compatible
            CExtensibleEnum(E e) : m_id(static_cast<int>(e)) {}

            // Explicit construction from raw int — for extended values
            explicit CExtensibleEnum(int id) : m_id(id) {}

            // Convert back to base enum — only valid for known values
            E asBaseEnum() const
            {
                if (!isBaseValue())
                    throw std::runtime_error("Not a base enum value: " + std::to_string(m_id));

                return static_cast<E>(m_id);
            }

            bool isBaseValue() const
            {
                return baseIds().count(m_id) > 0;
            }

            int id() const
            {
                return m_id;
            }

            std::string typeName() const
            {
                auto it = allNames().find(m_id);
                return it != allNames().end() ? it->second : "unknown(" + std::to_string(m_id) + ")";
            }

            std::string displayName() const
            {
                auto it = allDisplayNames().find(m_id);
                return it != allDisplayNames().end() ? it->second : "unknown(" + std::to_string(m_id) + ")";
            }

            bool operator==(const CExtensibleEnum& o) const { return m_id == o.m_id; }
            bool operator!=(const CExtensibleEnum& o) const { return m_id != o.m_id; }
            bool operator<(const CExtensibleEnum& o)  const { return m_id < o.m_id; }

            // Register a new extended value — intended to be called from Python
            static CExtensibleEnum registerExtended(const std::string& name, int id, const std::string& displayName)
            {
                if (allNames().count(id))
                    throw std::runtime_error("ID already registered: " + std::to_string(id));

                allNames()[id] = name;
                allDisplayNames()[id] = displayName;
                return CExtensibleEnum(id);
            }

            // Called once at startup to seed the registry from the base enum
            static void registerBase(int id, const std::string& name, const std::string& displayName)
            {
                baseIdsStorage().insert(id);
                allNames()[id] = name;
                allDisplayNames()[id] = displayName;
            }

        private:

            static void ensureRegistered()
            {
                static bool done = []() {
                    for (auto& [e, name, displayName] : EnumTraits<E>::values)
                        registerBase(static_cast<int>(e), name, displayName);

                    return true;
                }();

                (void)done;
            }

            // Raw storage — no ensureRegistered() call, safe to use during initialization
            static std::unordered_set<int>& baseIdsStorage()
            {
                static std::unordered_set<int> instance;
                return instance;
            }

            static std::unordered_set<int>& baseIds()
            {
                ensureRegistered();
                return baseIdsStorage();
            }

            static std::unordered_map<int, std::string>& allNames()
            {
                static std::unordered_map<int, std::string> instance;
                return instance;
            }

            static std::unordered_map<int, std::string>& allDisplayNames()
            {
                static std::unordered_map<int, std::string> instance;
                return instance;
            }

        private:

            int m_id;
    };
}

#endif // CEXTENSIBLEENUM_HPP
