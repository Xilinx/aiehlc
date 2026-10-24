/* lwIP options for the netboot supervisor (A78, standalone, SDT BSP).
 * Derived from the aiebaremetal example_lwip_echo configuration (RAW API,
 * NO_SYS, polled xemacif_input), with the TCP window raised for bulk ELF
 * transfer and DHCP removed (static IP on the lab board segment). */
#ifndef __LWIPOPTS_H_
#define __LWIPOPTS_H_

#ifndef PROCESSOR_LITTLE_ENDIAN
#define PROCESSOR_LITTLE_ENDIAN
#endif

#define SYS_LIGHTWEIGHT_PROT 1
#define NO_SYS 1
#define NO_SYS_NO_TIMERS 1
#define LWIP_SOCKET 0
#define LWIP_COMPAT_SOCKETS 0
#define LWIP_NETCONN 0
#define LWIP_TCP_KEEPALIVE 0

#define MEM_ALIGNMENT 64
#define MEM_SIZE (1024 * 1024)
#define MEMP_NUM_PBUF 64
#define MEMP_NUM_UDP_PCB 4
#define MEMP_NUM_TCP_PCB 8
#define MEMP_NUM_TCP_PCB_LISTEN 2
#define MEMP_NUM_TCP_SEG 512
#define MEMP_NUM_SYS_TIMEOUT 8
#define PBUF_POOL_SIZE 2048
#define PBUF_POOL_BUFSIZE 1700
#define PBUF_LINK_HLEN 16

#define ARP_TABLE_SIZE 10
#define ARP_QUEUEING 1
#define ICMP_TTL 255
#define IP_OPTIONS 0
#define IP_FORWARD 0
#define IP_REASSEMBLY 1
#define IP_FRAG 1
#define IP_REASS_MAX_PBUFS 128
#define IP_FRAG_MAX_MTU 1500
#define IP_DEFAULT_TTL 255
#define LWIP_CHKSUM_ALGORITHM 3

#define LWIP_UDP 1
#define UDP_TTL 255
#define LWIP_TCP 1
#define TCP_MSS 1460
/* 0xFFFF is the largest window without window scaling; the reference's 2048
 * would cap a 12.5 MB upload at a few hundred KB/s. */
#define TCP_WND 0xFFFF
#define TCP_SND_BUF (16 * TCP_MSS)
#define TCP_SND_QUEUELEN (4 * TCP_SND_BUF / TCP_MSS)
#define TCP_TTL 255
#define TCP_MAXRTX 12
#define TCP_SYNMAXRTX 4
#define TCP_QUEUE_OOSEQ 1
#define TCP_OVERSIZE TCP_MSS

/* GEM does the checksums. */
#define CHECKSUM_GEN_TCP 0
#define CHECKSUM_GEN_UDP 0
#define CHECKSUM_GEN_IP 0
#define CHECKSUM_CHECK_TCP 0
#define CHECKSUM_CHECK_UDP 0
#define CHECKSUM_CHECK_IP 0
#define LWIP_FULL_CSUM_OFFLOAD_RX 1
#define LWIP_FULL_CSUM_OFFLOAD_TX 1

#define MEMP_SEPARATE_POOLS 1
#define MEMP_NUM_FRAG_PBUF 256
#define IP_OPTIONS_ALLOWED 0

#define LWIP_DHCP 0
#define LWIP_ACD 0
#define CONFIG_LINKSPEED_AUTODETECT 1

#endif
