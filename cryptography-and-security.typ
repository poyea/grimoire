#import "template.typ": project

#project("Cryptography And Security")[
  #align(center)[#block(fill: luma(245), inset: 10pt, width: 80%, text(size: 9pt)[
    #align(center)[#text(size: 24pt, weight: "bold")[Cryptography And Security]]
  ])]

  #outline(title: "Table of Contents", depth: 2)

  #emph[Enjoy.]

  #pagebreak()

  #include "cryptography-and-security/symmetric-primitives.typ"
  #pagebreak()

  #include "cryptography-and-security/hashing-and-macs.typ"
  #pagebreak()

  #include "cryptography-and-security/asymmetric.typ"
  #pagebreak()

  #include "cryptography-and-security/digital-signatures.typ"
  #pagebreak()

  #include "cryptography-and-security/key-exchange-and-pki.typ"
  #pagebreak()

  #include "cryptography-and-security/tls.typ"
  #pagebreak()

  #include "cryptography-and-security/zero-knowledge-proofs.typ"
  #pagebreak()

  #include "cryptography-and-security/post-quantum.typ"
  #pagebreak()

  #include "cryptography-and-security/side-channel-attacks.typ"
  #pagebreak()

]
