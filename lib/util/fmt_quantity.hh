#ifndef UTIL_PRINT_QUANTITY_HH
#define UTIL_PRINT_QUANTITY_HH

#include <iostream>
#include <string>
#include <sstream>
#include <cinttypes>

template<typename T>
std::string fmt_quantity(T q) {
	constexpr char suffices[][4] = {"", "Ki", "Mi", "Gi"};
	constexpr double ranges[] = {1, 1 << 10, 1 << 20, 1 << 30};
	ptrdiff_t i = std::lower_bound(ranges,
			ranges + sizeof ranges / sizeof ranges[0], double(q)) - ranges - 1;
	std::ostringstream out;
	out.precision(3);
	out << (q/ranges[i]) << suffices[i];
	return out.str();
}

#endif // UTIL_PRINT_QUANTITY_HH
