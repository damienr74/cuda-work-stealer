#ifndef CUDA_CHECK_MACRO
#define CUDA_CHECK_MACRO

#include <iostream>

class SinkBuffer : public std::streambuf {
public:
	int overflow(int c) { return c; }
};

struct CudaErrorHandler {
	const cudaError_t err;
	SinkBuffer sink;
	std::ostream sinkStream;
	std::ostream& stream;

	CudaErrorHandler(cudaError_t err, std::ostream& stream)
		: err{err}
		, sink{}
		, sinkStream{&sink}
		, stream{err == cudaSuccess? sinkStream : stream}
		{}
	
	auto operator()(const char *const file,
		const char *const func,
		int line) -> std::ostream& {
		

		return stream << file << ':'
			<< func << ':'
			<< line << ": cudaError_t("
			<< cudaGetErrorString(err) << "): ";
	}
};

#define CUDA_CHECK(errExpr) \
		CudaErrorHandler((errExpr), std::cout)(__FILE__, __func__, __LINE__)


#endif // CUDA_CHECK_MACRO
