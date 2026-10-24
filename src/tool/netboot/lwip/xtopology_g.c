/* What the Vitis BSP generator would emit for this board (SDT form): one GEM,
 * terminated by a zero base address. */
#include "netif/xtopology.h"
#include "xparameters.h"

struct xtopology_t xtopology[] = {
    {XPAR_XEMACPS_0_BASEADDR, xemac_type_emacps},
    {0, xemac_type_unknown},
};

int xtopology_n_emacs = 1;
