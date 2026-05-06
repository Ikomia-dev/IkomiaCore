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

            // Called only from ensureRegistered() — must use *Storage() accessors exclusively.
            static void registerBase(int id, const std::string& name, const std::string& displayName)
            {
                baseIdsStorage().insert(id);
                allNamesStorage()[id] = name;
                allDisplayNamesStorage()[id] = displayName;
                nameToIdStorage()[name] = id;
            }

            // Register a new extended value — intended to be called from Python
            static CExtensibleEnum registerExtended(const std::string& name, int id, const std::string& displayName)
            {
                if (allNames().count(id))
                    throw std::runtime_error("ID already registered: " + std::to_string(id));

                allNames()[id] = name;
                allDisplayNames()[id] = displayName;
                nameToId()[name] = id;
                return CExtensibleEnum(id);
            }

            static CExtensibleEnum fromTypeName(const std::string& typeName)
            {
                auto it = nameToId().find(typeName);
                if (it == nameToId().end())
                    throw std::runtime_error("Unknown type name: " + typeName);
                return CExtensibleEnum(it->second);
            }

        private:

            static void ensureRegistered()
            {
                // Uses only *Storage() accessors — no re-entrancy risk.
                static bool done = []() {
                    for (auto& [e, name, displayName] : EnumTraits<E>::values)
                        registerBase(static_cast<int>(e), name, displayName);

                    return true;
                }();

                (void)done;
            }

            // Raw storage accessors — no ensureRegistered() call, safe to use during initialization.
            // registerBase() must use these exclusively to avoid re-entrant static initialization (UB).
            static std::unordered_set<int>& baseIdsStorage()
            {
                static std::unordered_set<int> instance;
                return instance;
            }

            static std::unordered_map<int, std::string>& allNamesStorage()
            {
                static std::unordered_map<int, std::string> instance;
                return instance;
            }

            static std::unordered_map<int, std::string>& allDisplayNamesStorage()
            {
                static std::unordered_map<int, std::string> instance;
                return instance;
            }

            static std::unordered_map<std::string, int>& nameToIdStorage()
            {
                static std::unordered_map<std::string, int> instance;
                return instance;
            }

            // Public accessors — trigger lazy registration on first use.
            static std::unordered_set<int>& baseIds()
            {
                ensureRegistered();
                return baseIdsStorage();
            }

            static std::unordered_map<int, std::string>& allNames()
            {
                ensureRegistered();
                return allNamesStorage();
            }

            static std::unordered_map<int, std::string>& allDisplayNames()
            {
                ensureRegistered();
                return allDisplayNamesStorage();
            }

            static std::unordered_map<std::string, int>& nameToId()
            {
                ensureRegistered();
                return nameToIdStorage();
            }

        private:

            int m_id;
    };
}

#endif // CEXTENSIBLEENUM_HPP
