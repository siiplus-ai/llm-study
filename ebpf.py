from bcc import BPF

bpf_src="""
int kprobe__sys_clone(void *ctx)
{
    bpf_trace_printk("Hello, World!\\n");
    return 0;
}
"""

b = BPF(text=bpf_src)
b.trace_print()