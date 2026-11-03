#import "../template.typ": xref

= Bootloaders, Secure Boot, and Firmware Updates <bootloaders-and-updates>

Every connected device will need new firmware in the field: to fix bugs, to patch vulnerabilities, to meet regulations that now require security updates. An update mechanism has to satisfy three demands at once. It must never brick the device, even if power fails halfway through writing flash. It must run only firmware the manufacturer signed. And it must not let an attacker roll the device back to an older, vulnerable version. This chapter covers the boot chain of a microcontroller, the flash layouts that make updates atomic, the cryptographic chain of trust, and the transport and manifest formats that carry updates to devices.

*See also:* #xref("embedded-and-realtime", "linker-scripts-and-startup", label: "Linker Scripts, Startup Code, and the Memory Map"), #xref("embedded-and-realtime", "mcus-and-soc", label: "Microcontrollers and Systems-on-Chip"), #xref("cryptography-and-security", "digital-signatures", label: "Digital Signatures") (cryptography), #xref("operating-systems", "boot-and-init", label: "Boot and Init") (operating systems)

== The Boot Chain

A typical secure $"MCU"$ boots in three stages:

1. *Boot ROM.* Mask ROM written at the factory, impossible to change. It reads boot configuration (option bytes, fuses, boot pins), may offer a recovery loader over $"UART"$ or USB, and on secure parts verifies the next stage against a key hash burned into one-time-programmable (OTP) memory.
2. *First-stage bootloader* in protected flash. It is small (16 to 64 KB), rarely updated, and owned by the manufacturer. It decides which application image to run, verifies it, applies pending updates, and handles rollback. MCUboot is the de facto open-source choice, used by Zephyr, Trusted Firmware-M, Mynewt and many vendor SDKs.
3. *Application.* The firmware that does the product's job, including the code that downloads new images.

Each stage verifies the next before running it. The ROM is the *root of trust*: if it is correct and its key hash cannot be changed, every later stage can be traced back to it.

=== Handing Over to the Application

The jump from bootloader to application must leave the processor as if it had just come out of reset, because the application's startup code assumes reset state. Stale interrupts, a running SysTick, or a vector table still pointing at the bootloader are classic sources of crashes in the first milliseconds of the application.

```cpp
#include <cstdint>

namespace regs {
inline volatile std::uint32_t& vtor     = *reinterpret_cast<volatile std::uint32_t*>(0xE000'ED08u);
inline volatile std::uint32_t& syst_csr = *reinterpret_cast<volatile std::uint32_t*>(0xE000'E010u);
inline volatile std::uint32_t* const nvic_icer = reinterpret_cast<volatile std::uint32_t*>(0xE000'E180u);
inline volatile std::uint32_t* const nvic_icpr = reinterpret_cast<volatile std::uint32_t*>(0xE000'E280u);
}

[[noreturn]] void jump_to_application(std::uintptr_t image_base) {
    const auto* vectors = reinterpret_cast<const std::uint32_t*>(image_base);
    const std::uint32_t initial_sp = vectors[0];
    const std::uint32_t reset_vector = vectors[1];

    asm volatile("cpsid i" ::: "memory");

    regs::syst_csr = 0;                    // stop SysTick
    for (int i = 0; i < 8; ++i) {          // 8 x 32 = 256 interrupt lines
        regs::nvic_icer[i] = 0xFFFF'FFFFu; // disable
        regs::nvic_icpr[i] = 0xFFFF'FFFFu; // clear pending
    }
    // Also return every peripheral and clock the bootloader used to its
    // reset state (vendor RCC reset registers), or the application inherits them.

    regs::vtor = image_base;               // application vector table
    asm volatile("dsb\n\tisb" ::: "memory");

    asm volatile(
        "msr msp, %0\n\t"
        "bx  %1\n\t"
        :: "r"(initial_sp), "r"(reset_vector) : "memory");
    __builtin_unreachable();
}
```

Interrupts remain masked by `PRIMASK` across the jump, so the application's startup code must execute `cpsie i` when it is ready. The reset vector has bit 0 set (Thumb state), which `BX` requires. On Armv8-M with TrustZone the hand-over is more involved, because the secure world (often TF-M) stays resident and the jump goes to the non-secure image.

== Flash Layouts for Atomic Updates

Writing a new image over the only copy of the running firmware is the one design that guarantees bricked devices: a power cut or a corrupt download mid-write leaves nothing bootable. Every robust scheme keeps a known-good image until the new one has proved itself.

#table(
  columns: 4,
  [*Scheme*], [*Flash needed*], [*Rollback*], [*Notes*],
  [Single slot, overwrite], [1x image], [None], [Only with a recovery loader in ROM and a human to use it],
  [A/B, execute in place from either], [2x image], [Instant: boot the other slot], [Images must be position-independent or linked twice; or use hardware bank swap],
  [Primary plus secondary, swap], [2x image plus scratch or one spare sector], [Swap back], [MCUboot's default; image always runs from the primary slot],
  [Primary plus secondary, overwrite only], [2x image], [None after copy], [Simpler and faster; needs another recovery path],
  [Compressed or delta into a staging area], [1x image plus staging], [Depends on scheme], [Saves flash; decompression must be resumable],
)

=== MCUboot's Swap with Test and Confirm

MCUboot's swap modes run the image from a fixed *primary slot* and receive downloads into a *secondary slot*. An update proceeds in steps:

1. The application downloads the new image into the secondary slot and marks it *pending, test* by writing a magic value into the slot's trailer.
2. On the next boot, MCUboot verifies the secondary image's signature and swaps the two slots sector by sector. *Swap-using-scratch* copies through a scratch sector. *Swap-using-move* first shifts the primary image up by one sector and then interleaves, avoiding the wear concentrated on a scratch sector. Progress is recorded in the trailer after each step, so a power cut resumes the swap from where it stopped.
3. The new image boots. If its self-test passes (it connected to the server, its sensors respond, its watchdog is being fed), it writes *confirmed* to its trailer.
4. If the device resets before the image confirms itself (a crash, a watchdog reset, a power cycle), MCUboot sees an unconfirmed image in the primary slot and swaps back to the previous one.

The confirmation step is the important design decision. Confirm too early (first thing in `main`) and an image that crashes after an hour, or that can no longer reach the update server, becomes permanent. Confirm only after the device has shown it can still do the one thing that matters most: receive the next update.

=== Hardware Dual-Bank

Many parts (STM32 dual-bank devices, NXP parts with flash remap, RP2350 with partition tables) can swap the address mapping of two flash banks with one option-byte write. The new image is written to the inactive bank while the old one keeps running from the active bank, which also avoids stalling execution during flash programming, and the bank swap at reboot is atomic. The image must be linked for the active bank's address, which is the same for both banks after the swap.

=== Flash Wear and Erase Granularity

NOR flash is erased in sectors (2 KB to 128 KB on $"MCU"$ flash) and endures roughly 10 000 to 100 000 erase cycles. An update scheme that writes a status word into the same sector on every boot can wear it out within the product's life. MCUboot keeps its state in the image trailers and writes it rarely. Application data that changes often (counters, logs, settings) belongs in a wear-levelled store such as Zephyr's NVS or littlefs, never in the image slots.

== Secure Boot

Secure boot means each stage refuses to run an image that does not carry a valid signature from a key it trusts.

- *Signing.* The build server computes a hash (SHA-256) over the image header and body and signs it with a private key that never leaves a hardware security module. MCUboot supports ECDSA P-256, Ed25519 and RSA. Elliptic-curve signatures are preferred on $"MCU"$s because verification is fast (tens of milliseconds on a Cortex-M4 without acceleration, much less with a crypto engine) and the keys are small.
- *Key storage.* The bootloader holds the public key, or the device holds a hash of it in OTP fuses, and the ROM checks that the bootloader's embedded key matches. The public key is not secret, but it must be *immutable*: an attacker who can replace it can sign anything.
- *Key revocation.* Fuse-based designs reserve several key slots and a revocation bitmap, so a compromised signing key can be retired by an update that burns a revocation fuse.
- *Protecting the bootloader.* Write-protect its flash sectors, and enable the vendor's readout and debug protection (STM32 RDP level 2 permanently disables the debug port, for example). A secure boot chain is worthless if a debugger can rewrite the first stage.

=== Anti-Rollback

A validly signed old image with a known vulnerability is still validly signed. To stop an attacker installing it, each image carries a *security counter*, and the device stores the highest counter it has accepted in a monotonic store: OTP fuses (truly irreversible but limited in number), or a counter protected by a secure element or TrustZone. The bootloader refuses any image whose counter is below the stored value, and raises the stored value only after the new image has been *confirmed*. Raising it at download time would make rollback to the previous, working image impossible. The security counter is distinct from the version number: only security fixes need to bump it, so ordinary downgrades for debugging remain possible.

=== Confidentiality

Signing proves origin but does not hide the firmware. If the image contains secrets or valuable intellectual property, it is also encrypted, typically with AES in counter mode, using a per-image key wrapped with a device or fleet key (MCUboot supports ECIES-P256 and ECIES-X25519 key encapsulation and AES-KW). Decryption happens during the swap, so the plaintext exists only inside the device's internal flash.

== Delta Updates and Compression

Over a cellular or LoRa link, sending a full 512 KB image to a million devices is expensive and slow. *Delta updates* send only the difference between the old and new images: tools such as bsdiff, detools and janpatch generate a patch, and the device rebuilds the new image from the old one plus the patch. Compiler output is unstable (moving one function shifts every later address), so deltas are often 10 to 40 percent of the full image rather than the 1 percent the source change suggests. Patch application must be resumable after a power failure, and must have the exact expected old image as input, so the server needs to know precisely what each device is running.

== Transport and Manifests

The bootloader verifies and installs; something else must deliver the image:

- *Wired*: USB DFU (Device Firmware Upgrade class), a $"UART"$ protocol in the bootloader or ROM, or the debug port in production.
- *Local wireless*: the SMP protocol (mcumgr) over Bluetooth Low Energy, used by Zephyr and Nordic devices.
- *Cloud OTA*: the device polls or is notified, downloads over HTTPS, CoAP or MQTT, verifies, and stores into the secondary slot. Fleet services stage rollouts (1 percent, 10 percent, everyone) and stop them automatically if devices on the new version stop reporting.

A *manifest* describes an update: which device classes it applies to, which version it requires as a starting point, where the payload is, its digest, and what to do with it. The IETF SUIT working group standardises this for constrained devices: RFC 9019 defines the architecture and RFC 9124 the information model, with a CBOR-encoded, COSE-signed manifest format that a small bootloader can parse. Signing the manifest rather than only the image lets the server bind the image to conditions, such as "only for hardware revision C, only from version 2.3 or later".

Regulation has made all of this mandatory in some markets. UN Regulation No. 156 requires a certified software update management system for new vehicle types in participating countries, and the EU Cyber Resilience Act (2024) requires manufacturers of connected products to provide security updates for the expected product lifetime.

== Failure Modes to Design Against

#table(
  columns: 2,
  [*Failure*], [*Mitigation*],
  [Power loss during flash write], [Never write over the only good image; resumable swaps; status in trailers],
  [Image that boots but cannot update], [Confirm only after update connectivity is proven; watchdog-driven revert],
  [Corrupt download], [Digest check before marking pending; signature check in the bootloader],
  [Malicious image], [Signature verification against an immutable key],
  [Downgrade to a vulnerable image], [Monotonic security counter, raised only after confirmation],
  [Bootloader bug], [Keep it small and well tested; reserve a ROM recovery path; some designs allow an A/B bootloader update],
  [Wrong image for the hardware], [Manifest conditions on device class and hardware revision],
  [Fleet-wide bad release], [Staged rollout with automatic halt on health metrics],
)

== Further Reading

MCUboot Project (2024). "MCUboot Design Documentation." mcu-tools documentation.

Moran, B., Tschofenig, H., Brown, D., Meriac, M. (2021). "A Firmware Update Architecture for Internet of Things." RFC 9019.

Moran, B., Tschofenig, H., Birkholz, H. (2022). "A Manifest Information Model for Firmware Updates in Internet of Things (IoT) Devices." RFC 9124.

Arm (2024). "Platform Security Architecture: Trusted Boot and Firmware Update." Arm DEN 0072 specification.

Trusted Firmware (2024). "Trusted Firmware-M Documentation." TF-M reference.

UNECE (2021). "UN Regulation No. 156: Software Update and Software Update Management System." Standard.

European Parliament and Council (2024). "Regulation (EU) 2024/2847 (Cyber Resilience Act)." Official Journal of the European Union.

Percival, C. (2003). "Naive Differences of Executable Code." bsdiff technical report.
