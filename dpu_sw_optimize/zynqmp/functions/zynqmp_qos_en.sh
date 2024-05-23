#!/usr/bin/env bash

# MPSOC chip only

#debug
VERBOSE="yes"

PATH="/usr/local/sbin:/usr/local/bin:/sbin:/bin:/usr/sbin:/usr/bini:$PATH"

[ ! -x /sbin/devmem ] && {echo "could not find devmem, exit" && exit 0}

qos_msg() {
    test "$VERBOSE" != "no" && printf "[%s]  0x%x:    0x%x => " "$1" $2 $(/sbin/devmem $2 32)
}

qos_dbg() {
    test "$VERBOSE" != "no" && printf "0x%x\n" $(/sbin/devmem $1 32)
}

axi_port_classfication()
{
    # UG1087:DDR_QOS_CTRL Module
    ddr_qos_ctrl=0xfd090000
    test "$VERBOSE" != no && printf "\nDDR_PORT_TYPE\n=====================================\n"
    qos_msg "PORT_TYPE" "$ddr_qos_ctrl"
    # Set port type (0xa845), shown as following:
    # port5-type: [15:14]--2--video traffic
    # port4-type: [13:12]--2--video traffic
    # port3-type: [11:10]--2--video traffic
    # port2B-type: [9:8] --0--best effort
    # port2R-type: [7:6] --1--low  latency
    # port1B-type: [5:4] --0--best effort
    # port1R-type: [3:2] --1--low  latency
    # port0-type: [1:0] --1-- low  latency
    devmem "$ddr_qos_ctrl" 32 0xa845
    qos_dbg "$ddr_qos_ctrl"
}

afi_hp_ddrc_qos()
{
    local hp0_rdctrl=0xfd380000
    local hp0_rdissue=0xfd380004
    local hp0_rdqos=0xfd380008
    local hp0_wrctrl=0xfd380014
    local hp0_wrissue=0xfd380018
    local hp0_wrqos=0xfd38001c

    for ((i=0;i<4;i++))
    do
        test "$VERBOSE" != no && printf "\nS_AXI_HP%d_FPD (0-low f-high)\n=====================================\n" $i
    
        # Check if FABRIC_QOS_EN-bit[2]=0 ?
        # The QoS bits are derived from APB register, AFIFM_RD/WRQoS.staticQoS
        ((hp_rdctrl=$hp0_rdctrl+$i*0x10000))
        qos_msg "RDCTRL " "$hp_rdctrl"
        qos_dbg "$hp_rdctrl"
        ((hp_wrctrl=$hp0_wrctrl+$i*0x10000))
        qos_msg "WRCTRL " "$hp_wrctrl"
        qos_dbg "$hp_wrctrl"
    
        ((hp_rdissue=$hp0_rdissue+$i*0x10000))
        qos_msg "RDISSUE" "$hp_rdissue"
        devmem "$hp_rdissue" 32 $1
        qos_dbg "$hp_rdissue"
    
        ((hp_rdqos=$hp0_rdqos+$i*0x10000))
        qos_msg "RDQoS  " "$hp_rdqos"
        devmem "$hp_rdqos" 32 $2
        qos_dbg "$hp_rdqos"
    
        ((hp_wrissue=$hp0_wrissue+$i*0x10000))
        qos_msg "WRISSUE" "$hp_wrissue"
        devmem "$hp_wrissue" 32 $3
        qos_dbg "$hp_wrissue"
    
        ((hp_wrqos=$hp0_wrqos+$i*0x10000))
        qos_msg "WRQoS  " "$hp_wrqos"
        devmem "$hp_wrqos" 32 $4
        qos_dbg "$hp_wrqos"
    done
}

afi_hpc0_ddrc_qos()
{
    local hpc0_rdissue=0xfd360004
    local hpc0_rdqos=0xfd360008
    local hpc0_wrissue=0xfd360018
    local hpc0_wrqos=0xfd36001c
    test "$VERBOSE" != no && printf "\nS_AXI_HPC0_FPD (0-low f-high)\n=====================================\n"

    qos_msg "RDISSUE" "$hpc0_rdissue"
    devmem "$hpc0_rdissue" 32 $1
    qos_dbg "$hpc0_rdissue"

    qos_msg "RDQoS  " "$hpc0_rdqos"
    devmem "$hpc0_rdqos" 32 $2
    qos_dbg "$hpc0_rdqos"

    qos_msg "WRISSUE" "$hpc0_wrissue"
    devmem "$hpc0_wrissue" 32 $3
    qos_dbg "$hpc0_wrissue"

    qos_msg "WRQoS  " "$hpc0_wrqos"
    devmem "$hpc0_wrqos" 32 $4
    qos_dbg "$hpc0_wrqos"
}

afi_hpc1_ddrc_qos()
{
    local hpc1_rdissue=0xfd370004
    local hpc1_rdqos=0xfd370008
    local hpc1_wrissue=0xfd370018
    local hpc1_wrqos=0xfd37001c
    test "$VERBOSE" != no && printf "\nS_AXI_HPC1_FPD (0-low f-high)\n=====================================\n"

    qos_msg "RDISSUE" "$hpc1_rdissue"
    devmem "$hpc1_rdissue" 32 $1
    qos_dbg "$hpc1_rdissue"

    qos_msg "RDQoS  " "$hpc1_rdqos"
    devmem "$hpc1_rdqos" 32 $2
    qos_dbg "$hpc1_rdqos"

    qos_msg "WRISSUE" "$hpc1_wrissue"
    devmem "$hpc1_wrissue" 32 $3
    qos_dbg "$hpc1_wrissue"

    qos_msg "WRQoS  " "$hpc1_wrqos"
    devmem "$hpc1_wrqos" 32 $4
    qos_dbg "$hpc1_wrqos"
}

qos_config()
{
    axi_port_classfication

    # increase the read issuing capability of the port connected to DPU.
    # By default, it can take a maximum of 8 requests at a time, and increasing
    # the issueing capability may keep the ports busy with always some requests in the queue.
    # Set RDISSUE-AFIFM register to allow 16 requests at a time, leave others as it is
    #
    # incoming ARQOS values 
    # 0-3: LPR(Low priority request) for "BE"(best effort) traffic type
    # 4+ : for Video traffic type.
    #
    # Here traffic type sets to "BE"

    # hpc0 --- port1 on DDR controller
    afi_hpc0_ddrc_qos 15 0 15 0

    # hpc1 --- port2 on DDR controller
    afi_hpc1_ddrc_qos 15 0 15 0

    # hp0 --- port3 --- hp0 share port3 with DisplayPort
    # hp1 --- port4
    # hp2 --- port4
    # hp3 --- port5
    afi_hp_ddrc_qos   15 0 15 0

    return 0
}
