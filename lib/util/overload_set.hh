#ifndef UTIL_OVERLOAD_SET_HH
#define UTIL_OVERLOAD_SET_HH

template<class ...Fs> struct OverloadSet : Fs... { using Fs::operator()...; };
template<class ...Fs> OverloadSet(Fs...) -> OverloadSet<Fs...>;

#endif // UTIL_OVERLOAD_SET_HH
