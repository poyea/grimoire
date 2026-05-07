= Kernel Modules

A kernel module is code loaded into the running kernel at runtime, sharing its address space and privilege level. Modules are how Linux ships drivers, filesystems, network protocols, and crypto algorithms without baking them into a monolithic image. Writing one is straightforward; writing one *correctly* requires care, because everything the module does runs in ring 0 and a bug typically panics the box.

This chapter is a working-engineer's guide: a minimal module, the major device-class APIs, and the deployment story (signing, DKMS, blacklisting).

== A Minimal Module

```c
// hello.c
#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>

static int __init hello_init(void) {
    pr_info("hello: loaded\n");
    return 0;
}

static void __exit hello_exit(void) {
    pr_info("hello: unloaded\n");
}

module_init(hello_init);
module_exit(hello_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Your Name");
MODULE_DESCRIPTION("Smallest possible kernel module");
MODULE_VERSION("0.1");
```

```makefile
# Makefile
obj-m += hello.o

all:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean
```

Build, load, observe, unload:

```
$ make
$ sudo insmod ./hello.ko
$ dmesg | tail -1
[12345.678] hello: loaded
$ sudo rmmod hello
$ dmesg | tail -1
[12356.789] hello: unloaded
```

*`MODULE_LICENSE`* is not decoration — modules without a GPL-compatible license are flagged "tainted" and lose access to GPL-only kernel symbols (`EXPORT_SYMBOL_GPL`-marked functions). Most useful kernel APIs are GPL-only.

*The `__init` and `__exit` annotations* tell the linker to place this code in special sections that are freed after init / never linked into a built-in module. Use them religiously.

== Module Parameters

```c
static int debug_level = 0;
module_param(debug_level, int, 0644);
MODULE_PARM_DESC(debug_level, "Verbosity 0-3");

static char *device_name = "default";
module_param(device_name, charp, 0644);
```

Set at load time:

```
sudo insmod ./hello.ko debug_level=2 device_name=my0
```

Or via modprobe + `/etc/modprobe.d/foo.conf`:

```
options hello debug_level=2 device_name=my0
```

The third arg to `module_param` is the file mode of the corresponding `/sys/module/hello/parameters/<name>` entry. `0644` makes it readable; setting the write bit lets root change the value at runtime.

== Logging: pr_\* and dev_\*

```c
pr_emerg(...)      pr_alert(...)     pr_crit(...)      pr_err(...)
pr_warn(...)       pr_notice(...)    pr_info(...)      pr_debug(...)

// With a struct device:
dev_err(&pdev->dev, "i2c xfer failed: %d\n", err);
```

`pr_debug` calls disappear at compile time unless `DEBUG` is defined, or are dynamically toggled via dynamic-debug:

```
echo 'file hello.c +p' > /sys/kernel/debug/dynamic_debug/control
```

== Character Devices

The classical "device file" abstraction — a node in `/dev` that user-space opens, reads, writes, and ioctls. Block devices (disks) and network devices (sockets) have their own subsystems; everything else (sensors, simple peripherals, virtual devices) is a character device.

```c
#include <linux/cdev.h>
#include <linux/fs.h>

static dev_t mydev_devt;
static struct cdev mydev_cdev;
static struct class *mydev_class;

static int mydev_open(struct inode *inode, struct file *file) { return 0; }
static int mydev_release(struct inode *inode, struct file *file) { return 0; }

static ssize_t mydev_read(struct file *file, char __user *buf,
                          size_t count, loff_t *off) {
    char *msg = "hello\n";
    size_t len = strlen(msg);
    if (*off >= len) return 0;
    if (count > len - *off) count = len - *off;
    if (copy_to_user(buf, msg + *off, count)) return -EFAULT;
    *off += count;
    return count;
}

static long mydev_ioctl(struct file *file, unsigned int cmd, unsigned long arg) {
    switch (cmd) {
    case MYDEV_RESET:
        do_reset();
        return 0;
    default:
        return -ENOTTY;
    }
}

static const struct file_operations mydev_fops = {
    .owner          = THIS_MODULE,
    .open           = mydev_open,
    .release        = mydev_release,
    .read           = mydev_read,
    .unlocked_ioctl = mydev_ioctl,
    .llseek         = no_llseek,
};

static int __init mydev_init(void) {
    int err = alloc_chrdev_region(&mydev_devt, 0, 1, "mydev");
    if (err) return err;

    cdev_init(&mydev_cdev, &mydev_fops);
    err = cdev_add(&mydev_cdev, mydev_devt, 1);
    if (err) goto fail_unreg;

    mydev_class = class_create("mydev");
    if (IS_ERR(mydev_class)) { err = PTR_ERR(mydev_class); goto fail_cdev; }
    device_create(mydev_class, NULL, mydev_devt, NULL, "mydev");

    return 0;

fail_cdev:
    cdev_del(&mydev_cdev);
fail_unreg:
    unregister_chrdev_region(mydev_devt, 1);
    return err;
}
```

`device_create` on `mydev_class` causes udev to create `/dev/mydev`. Without it, you would have to `mknod` manually.

*`copy_to_user` / `copy_from_user`*: the *only* safe way to move data across the user/kernel boundary. They handle page faults, validate the address range, and bypass aliasing pitfalls. Direct memcpy on a `__user` pointer is a bug.

*ioctl numbers* should be defined with the macros in `<linux/ioctl.h>` so they encode direction and size:

```c
#define MYDEV_IOC_MAGIC   'k'
#define MYDEV_RESET       _IO(MYDEV_IOC_MAGIC, 0)
#define MYDEV_GET_STATS   _IOR(MYDEV_IOC_MAGIC, 1, struct mydev_stats)
#define MYDEV_SET_CONFIG  _IOW(MYDEV_IOC_MAGIC, 2, struct mydev_config)
```

*`compat_ioctl`*: needed for 32-bit user-space on a 64-bit kernel, or vice versa, when struct layouts differ. If your structs are layout-compatible across word sizes, `compat_ioctl = compat_ptr_ioctl` suffices.

== procfs

`/proc/<name>` entries — virtual files generated by the kernel. Mostly used for diagnostic output.

```c
#include <linux/proc_fs.h>
#include <linux/seq_file.h>

static int mydev_show(struct seq_file *m, void *v) {
    seq_printf(m, "events: %llu\n", atomic64_read(&events));
    seq_printf(m, "errors: %llu\n", atomic64_read(&errors));
    return 0;
}
DEFINE_SHOW_ATTRIBUTE(mydev);

proc_create("mydev", 0444, NULL, &mydev_proc_ops);
```

`seq_file` handles the common pattern of "produce text on demand, support partial reads, support `lseek`." Use it for anything beyond a one-line value.

*Don't add new entries to `/proc` in modern drivers.* `procfs` is for the legacy / process-oriented interface. New per-driver state belongs in sysfs (or debugfs for debugging-only).

== sysfs

`/sys/class/<name>/...` and `/sys/module/<mod>/...` — one attribute per file, simple values, machine-friendly. The standard place for driver tunables and status.

```c
static ssize_t enabled_show(struct device *d, struct device_attribute *a, char *buf) {
    struct my_dev *dev = dev_get_drvdata(d);
    return sysfs_emit(buf, "%d\n", dev->enabled);
}

static ssize_t enabled_store(struct device *d, struct device_attribute *a,
                             const char *buf, size_t count) {
    struct my_dev *dev = dev_get_drvdata(d);
    int v;
    if (kstrtoint(buf, 0, &v)) return -EINVAL;
    dev->enabled = !!v;
    return count;
}

static DEVICE_ATTR_RW(enabled);

static struct attribute *mydev_attrs[] = {
    &dev_attr_enabled.attr,
    NULL,
};
ATTRIBUTE_GROUPS(mydev);
```

`sysfs_emit` is the modern replacement for `sprintf` in show-functions; it knows the buffer size limit (PAGE_SIZE).

*sysfs rules:*

- *One value per file.* If you need a struct, you need multiple files.
- *No size limit issues — buffer is always PAGE_SIZE.* If you have more, use seq_file via debugfs.
- *Stable ABI.* Anything in `/sys/class/<your-class>/` is treated as ABI by user-space.

== debugfs

For driver-internal debugging. No ABI commitment, can be removed at any time, and only present if the kernel was built with `CONFIG_DEBUG_FS=y`.

```c
#include <linux/debugfs.h>

static struct dentry *dbg_root;

dbg_root = debugfs_create_dir("mydev", NULL);
debugfs_create_u32("event_count", 0444, dbg_root, &event_count);
debugfs_create_x32("status_reg",  0444, dbg_root, &status_reg);
debugfs_create_file("regs", 0444, dbg_root, NULL, &regs_fops);
```

debugfs typically mounts at `/sys/kernel/debug/`.

== Netfilter Hooks

Netfilter is the kernel's packet-filtering framework — what `iptables` and `nftables` configure. Modules can register their own hooks at five points along the network stack: `PRE_ROUTING`, `LOCAL_IN`, `FORWARD`, `LOCAL_OUT`, `POST_ROUTING`.

```c
#include <linux/netfilter.h>
#include <linux/netfilter_ipv4.h>

static unsigned int my_hook(void *priv, struct sk_buff *skb,
                            const struct nf_hook_state *state) {
    struct iphdr *iph = ip_hdr(skb);
    if (iph->protocol == IPPROTO_ICMP)
        return NF_DROP;
    return NF_ACCEPT;
}

static struct nf_hook_ops my_ops = {
    .hook     = my_hook,
    .pf       = NFPROTO_IPV4,
    .hooknum  = NF_INET_PRE_ROUTING,
    .priority = NF_IP_PRI_FIRST,
};

nf_register_net_hook(&init_net, &my_ops);
```

For new code, prefer *XDP* (eXpress Data Path) or *TC* (traffic control) BPF hooks over netfilter — they run earlier in the receive path (XDP runs *before* skb allocation) and are programmable from user-space without a kernel module.

== Concurrency: Locks and RCU

The kernel is fully preemptive and SMP. Every shared data structure needs synchronization.

#table(columns: (auto, 1fr),
  [*Primitive*], [*Use*],
  [`spinlock_t`], [Short critical sections, atomic context (IRQ handlers, softirqs).],
  [`raw_spinlock_t`], [Same, but stays a real spinlock under PREEMPT_RT (where `spinlock_t` becomes sleepable).],
  [`mutex`], [Process context, sleepable. Single owner.],
  [`rwsem`], [Read-write semaphore, sleepable.],
  [`rwlock_t`], [Read-write spinlock. Largely deprecated in favor of RCU for read-heavy paths.],
  [`atomic_t` / `atomic64_t`], [Lock-free counters with explicit memory ordering.],
  [`percpu_counter`], [Sloppy fast counter — per-CPU buckets summed on read.],
  [RCU], [Read-mostly data structures. Readers wait for nothing; writers must wait for grace period.],
)

Spinlock variants:

```c
spin_lock(&lock);                  // basic; only safe if you know IRQs aren't an issue
spin_lock_bh(&lock);               // also disables softirqs
spin_lock_irq(&lock);              // also disables IRQs (assumes they were enabled)
spin_lock_irqsave(&lock, flags);   // saves/restores IRQ state — always-correct version
```

If a softirq might also take the lock, use `_bh`. If a hard IRQ might, use `_irqsave`. Wrong choice deadlocks.

== Module Signing and Loading

Modern distros require signed modules. The kernel verifies the signature against keys in its built-in keyring (the kernel's *system keyring*) plus the *machine owner key* keyring (MOK, used by Secure Boot enrollment).

For out-of-tree development, sign your module with the kernel's local key (built when `CONFIG_MODULE_SIG=y`):

```
/usr/src/linux-headers-$(uname -r)/scripts/sign-file sha256 \
    /var/lib/shim-signed/mok/MOK.priv \
    /var/lib/shim-signed/mok/MOK.der \
    hello.ko
```

Or disable signature enforcement (development boxes only):

```
echo 0 > /sys/kernel/security/lockdown                 # if Secure Boot is off
# Boot with module.sig_enforce=0 if signing is enforced.
```

`insmod` loads a single .ko by path. `modprobe` resolves the name through `depmod`-built dependency files and recursively loads dependencies. Production: use `modprobe`.

```
modprobe my_module                       # load
modprobe -r my_module                     # unload (if refcount allows)
lsmod                                     # currently loaded
modinfo my_module                         # version, params, deps
```

*Blacklisting:*

```
echo "blacklist nouveau" > /etc/modprobe.d/blacklist-nouveau.conf
update-initramfs -u                        # if loaded from initramfs
```

== DKMS: Building Across Kernel Versions

DKMS (Dynamic Kernel Module Support) automates rebuilding out-of-tree modules when a new kernel is installed. The classic example is NVIDIA's proprietary driver.

`/usr/src/<modname>-<ver>/dkms.conf`:

```
PACKAGE_NAME="my_module"
PACKAGE_VERSION="1.0"
BUILT_MODULE_NAME[0]="my_module"
DEST_MODULE_LOCATION[0]="/kernel/extra"
MAKE[0]="make -C ${kernel_source_dir} M=${dkms_tree}/${PACKAGE_NAME}/${PACKAGE_VERSION}/build modules"
CLEAN="make -C ${kernel_source_dir} M=${dkms_tree}/${PACKAGE_NAME}/${PACKAGE_VERSION}/build clean"
AUTOINSTALL="yes"
```

Then:

```
sudo dkms add -m my_module -v 1.0
sudo dkms install -m my_module -v 1.0
```

DKMS hooks into the apt/yum kernel-upgrade path and rebuilds against new headers automatically.

== Debugging

```
sudo dmesg -wH                                        # tail kernel log
sudo cat /proc/kallsyms | grep my_                    # exported symbols
sudo cat /sys/kernel/debug/dynamic_debug/control      # dyndbg sites
addr2line -e my_module.ko -f <addr>                    # decode oops backtrace
```

For an actual oops or panic:

1. Capture the full `dmesg` from before the crash if possible. Otherwise reboot, mount `/var/crash` (kdump), or read it from the IPMI/BMC serial console.
2. Look at the `RIP:` line — that's the faulting instruction.
3. `addr2line` against your `.ko` (built with `-g`) maps RIP to source.
4. The "Code:" hexdump and register dump give you the values at fault.

`kgdb` and `lockdep` are invaluable but out of scope here.

== When NOT to Write a Kernel Module

Most things you'd reach for a kernel module for in 2026 are better solved in user-space:

- *Tracing / observability:* eBPF.
- *Packet processing:* XDP/TC BPF, AF_XDP, DPDK.
- *Custom filesystems:* FUSE.
- *USB devices:* libusb in user-space, no kernel driver needed for most.
- *Per-process resource control:* cgroups + namespaces.

A kernel module is the right answer when you genuinely need ring 0 — a new bus driver, a new filesystem implementation, a new hardware accelerator integration, a feature with strict latency requirements that BPF can't meet. For everything else, the user-space tooling is now mature enough that staying out of the kernel is the right default.

*See also:* _interrupts.typ_ (`request_irq`, NAPI, threaded IRQs), _kernel-tracing.typ_ (eBPF as the alternative to writing a probe-style module), _abi-syscalls.typ_ (adding a new syscall is a special case of kernel modification).
