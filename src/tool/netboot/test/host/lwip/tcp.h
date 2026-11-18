/* Host stub of the lwIP RAW TCP API server.c uses (x86 unit test). The test
 * driver implements the functions: it plays the client by calling the recv
 * callback with pbuf chains and collects what the server writes. */
#ifndef LWIP_TCP_H_STUB
#define LWIP_TCP_H_STUB
#include <stddef.h>
#include <stdint.h>
typedef int8_t err_t;
typedef uint8_t u8_t;
typedef uint16_t u16_t;
#define ERR_OK 0
#define ERR_MEM -1
#define ERR_VAL -6
#define TCP_WRITE_FLAG_COPY 0x01
#define IPADDR_TYPE_ANY 46U
#define IP_ANY_TYPE NULL
struct pbuf {
    struct pbuf *next;
    void *payload;
    u16_t tot_len;
    u16_t len;
};
struct tcp_pcb {
    int closed;
};
typedef err_t (*tcp_recv_fn)(void *arg, struct tcp_pcb *tpcb, struct pbuf *p, err_t err);
typedef err_t (*tcp_accept_fn)(void *arg, struct tcp_pcb *newpcb, err_t err);
typedef void (*tcp_err_fn)(void *arg, err_t err);
struct tcp_pcb *tcp_new_ip_type(u8_t type);
err_t tcp_bind(struct tcp_pcb *pcb, const void *ipaddr, u16_t port);
struct tcp_pcb *tcp_listen(struct tcp_pcb *pcb);
void tcp_accept(struct tcp_pcb *pcb, tcp_accept_fn accept);
void tcp_arg(struct tcp_pcb *pcb, void *arg);
void tcp_recv(struct tcp_pcb *pcb, tcp_recv_fn recv);
void tcp_err(struct tcp_pcb *pcb, tcp_err_fn err);
err_t tcp_write(struct tcp_pcb *pcb, const void *dataptr, u16_t len, u8_t apiflags);
err_t tcp_output(struct tcp_pcb *pcb);
err_t tcp_close(struct tcp_pcb *pcb);
void tcp_abort(struct tcp_pcb *pcb);
void tcp_recved(struct tcp_pcb *pcb, u16_t len);
u8_t pbuf_free(struct pbuf *p);
#endif
