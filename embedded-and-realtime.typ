#import "template.typ": project

#project("Embedded and Real-Time Systems")[
  #align(center)[#block(fill: luma(245), inset: 10pt, width: 80%, text(size: 9pt)[
    #align(center)[#text(size: 24pt, weight: "bold")[Embedded And Realtime]]
  ])]

  #outline(title: "Table of Contents", depth: 2)

  #emph[Enjoy.]

  #pagebreak()

  #include "embedded-and-realtime/mcus-and-soc.typ"
  #pagebreak()

  #include "embedded-and-realtime/linker-scripts-and-startup.typ"
  #pagebreak()

  #include "embedded-and-realtime/embedded-cpp.typ"
  #pagebreak()

  #include "embedded-and-realtime/peripherals-and-drivers.typ"
  #pagebreak()

  #include "embedded-and-realtime/fieldbuses.typ"
  #pagebreak()

  #include "embedded-and-realtime/rtos.typ"
  #pagebreak()

  #include "embedded-and-realtime/timing-analysis.typ"
  #pagebreak()

  #include "embedded-and-realtime/low-power-design.typ"
  #pagebreak()

  #include "embedded-and-realtime/bootloaders-and-updates.typ"
  #pagebreak()

  #include "embedded-and-realtime/debugging-and-testing.typ"
  #pagebreak()

]
