from vllm.v1.libsmctrl import _load_libsmctrl

lib = _load_libsmctrl()
print("Loaded libsmctrl:", lib)
print("Has global mask:", hasattr(lib, "libsmctrl_set_global_mask"))
print("Has stream mask:", hasattr(lib, "libsmctrl_set_stream_mask"))