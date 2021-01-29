// Copyright (c) 2009-2011, Tor M. Aamodt, Tayler Hetherington
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice, this
// list of conditions and the following disclaimer in the documentation and/or
// other materials provided with the distribution.
// Neither the name of The University of British Columbia nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "gpu-cache.h"
#include "stat-tool.h"
#include <assert.h>
//Zu_Hao: bypass switch
#define MAX_DEFAULT_CACHE_SIZE_MULTIBLIER 4
bool use_bypass_L1cache = 1;
bool use_bypass_L2cache = 1;
bool is_use_L1VC = 1;   //VC開關

//Zu_Hao: those variables are used in L1 data cache or L1 victim cache 
unsigned L1_warp_div_match[5][34]={0}; 
unsigned L2_warp_div_match[5][34]={0}; 
unsigned L1_temp_div = 0;
unsigned L1_temp_miss_div =0;
int      L1_temp_ishit = 0;
unsigned L1_status_invaild=0;    //紀錄L1快取裡面要被替換的cache line是不是invalid 
unsigned L1_ishit_count[11] = {0};
int      L1VC_access =0;
int      L1VC_hit = 0;
int      L1VC_miss =0;
int      L1VC_RF =0;
unsigned L2_temp_div = 0;
unsigned L2_temp_miss_div =0;
int      L2_temp_ishit = 0;
unsigned L2_status_invaild=0;    //紀錄L1快取裡面要被替換的cache line是不是invalid 
unsigned L2_ishit_count[11] = {0};
int      L2VC_access =0;
int      L2VC_hit = 0;
int      L2VC_miss =0;
int      L2VC_RF =0;   
int      wb_div  = 0;
void cache_statistic::return_VC_statistic(){
    printf("L1VC_ACCESS = %d \n L1VC_HIT = %d \n L1VC_MISS = %d \n L1VC_RF = %d\n",L1VC_access,L1VC_hit,L1VC_miss,L1VC_RF);
    printf("L2VC_ACCESS = %d \n L2VC_HIT = %d \n L2VC_MISS = %d \n L2VC_RF = %d\n",L2VC_access,L2VC_hit,L2VC_miss,L2VC_RF);
}
void cache_statistic::return_div_match_statistic(){
    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 34; j++)
        {
            printf("Request %d hit %d = %d \n",i , j ,L1_warp_div_match[i][j]);
        }   
    }
    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 34; j++)
        {
            printf("L2Request %d hit %d = %d \n",i , j ,L2_warp_div_match[i][j]);
        }   
    }
}
// used to allocate memory that is large enough to adapt the changes in cache size across kernels
const char * cache_request_status_str(enum cache_request_status status) 
{
   static const char * static_cache_request_status_str[] = {
      "HIT",
      "HIT_RESERVED",
      "MISS",
      "RESERVATION_FAIL"
   }; 

   assert(sizeof(static_cache_request_status_str) / sizeof(const char*) == NUM_CACHE_REQUEST_STATUS); 
   assert(status < NUM_CACHE_REQUEST_STATUS); 

   return static_cache_request_status_str[status]; 
}

unsigned l1d_cache_config::set_index(new_addr_type addr) const{
    unsigned set_index = m_nset; // Default to linear set index function
    unsigned lower_xor = 0;
    unsigned upper_xor = 0;

    switch(m_set_index_function){
    case FERMI_HASH_SET_FUNCTION:
        /*
        * Set Indexing function from "A Detailed GPU Cache Model Based on Reuse Distance Theory"
        * Cedric Nugteren et al.
        * ISCA 2014
        */
        if(m_nset == 32 || m_nset == 64){
            // Lower xor value is bits 7-11
            lower_xor = (addr >> m_line_sz_log2) & 0x1F;

            // Upper xor value is bits 13, 14, 15, 17, and 19
            upper_xor  = (addr & 0xE000)  >> 13; // Bits 13, 14, 15
            upper_xor |= (addr & 0x20000) >> 14; // Bit 17
            upper_xor |= (addr & 0x80000) >> 15; // Bit 19

            set_index = (lower_xor ^ upper_xor);

            // 48KB cache prepends the set_index with bit 12
            if(m_nset == 64)
                set_index |= (addr & 0x1000) >> 7;

        }else{ /* Else incorrect number of sets for the hashing function */
            assert("\nGPGPU-Sim cache configuration error: The number of sets should be "
                    "32 or 64 for the hashing set index function.\n" && 0);
        }
        break;

    case CUSTOM_SET_FUNCTION:
        /* No custom set function implemented */
        break;

    case LINEAR_SET_FUNCTION:
        set_index = (addr >> m_line_sz_log2) & (m_nset-1);
        break;
    }

    // Linear function selected or custom set index function not implemented
    assert((set_index < m_nset) && "\nError: Set index out of bounds. This is caused by "
            "an incorrect or unimplemented custom set index function.\n");

    return set_index;
}

void l2_cache_config::init(linear_to_raw_address_translation *address_mapping){
    cache_config::init(m_config_string,FuncCachePreferNone);
    m_address_mapping = address_mapping;
}

unsigned l2_cache_config::set_index(new_addr_type addr) const{
    if(!m_address_mapping){
        return(addr >> m_line_sz_log2) & (m_nset-1);
    }else{
        // Calculate set index without memory partition bits to reduce set camping
        new_addr_type part_addr = m_address_mapping->partition_address(addr);
        return(part_addr >> m_line_sz_log2) & (m_nset -1);
    }
}

tag_array::~tag_array() 
{
    delete[] m_lines;
}

tag_array::tag_array( cache_config &config,
                      int core_id,
                      int type_id,
                      cache_block_t* new_lines)
    : m_config( config ),
      m_lines( new_lines )
{
    init( core_id, type_id );
}

void tag_array::update_cache_parameters(cache_config &config)
{
    m_config=config;
}

tag_array::tag_array( cache_config &config,
                      int core_id,
                      int type_id )
    : m_config( config )
{
    //assert( m_config.m_write_policy == READ_ONLY ); Old assert
    m_lines = new cache_block_t[MAX_DEFAULT_CACHE_SIZE_MULTIBLIER*config.get_num_lines()];
    init( core_id, type_id );
}

void tag_array::init( int core_id, int type_id )
{
    m_access = 0;
    m_miss = 0;
    m_pending_hit = 0;
    m_res_fail = 0;
    // initialize snapshot counters for visualizer
    m_prev_snapshot_access = 0;
    m_prev_snapshot_miss = 0;
    m_prev_snapshot_pending_hit = 0;
    m_core_id = core_id; 
    m_type_id = type_id;
}
//Zu_Hao:L1 cache probe_div_match
enum cache_request_status tag_array::probe_div_match( new_addr_type addr, unsigned &idx ,int  request_div) const {
    //assert( m_config.m_write_policy == READ_ONLY );
    L1_temp_ishit= -1;
    L1_temp_miss_div=-1;
    L1_status_invaild=0;
    unsigned set_index = m_config.set_index(addr);
    new_addr_type tag = m_config.tag(addr);        

    unsigned invalid_line = (unsigned)-1;
    unsigned valid_line = (unsigned)-1;
    unsigned valid_timestamp = (unsigned)-1;
    bool all_reserved = true;
    // check for hit or pending hit
    for (unsigned way=0; way<m_config.m_assoc; way++) {
        unsigned index = set_index*m_config.m_assoc+way;
        cache_block_t *line = &m_lines[index];
        if (line->m_tag == tag) {  
            if ( line->m_status == RESERVED ) {
                idx = index;
                return HIT_RESERVED;
            } else if ( line->m_status == VALID ) {
                idx = index;
                if ( (request_div>-1) && (request_div<5))
                {
                    if(line->block_div_num != request_div)
                    {
                        L1_warp_div_match[request_div][line->block_div_num]++;
                    }
                    else
                        L1_warp_div_match[request_div][request_div]++;
                }
                L1_temp_div = line->block_div_num;
                line->ishit++;
                return HIT;
            } else if ( line->m_status == MODIFIED ) {
                idx = index;
                if ((request_div > -1) && (request_div < 5))
                {
                    if(line->block_div_num != request_div)
                        L1_warp_div_match[request_div][line->block_div_num]++;
                    else
                        L1_warp_div_match[request_div][request_div]++;
                }
                L1_temp_div = line->block_div_num;
                line->ishit++;
                return HIT;
            } else {
                assert( line->m_status == INVALID );
            }
        }
        if (line->m_status != RESERVED) { 
            all_reserved = false;
            if (line->m_status == INVALID) { //如果有invalid就先選
                invalid_line = index;
            } else {                        //沒有invalid就用LRU  可能是MODIFY或是vaild
                if ( m_config.m_replacement_policy == LRU ) {
                    if ( line->m_last_access_time < valid_timestamp) {
                        valid_timestamp = line->m_last_access_time;
                        valid_line = index;
                    }
                } else if ( m_config.m_replacement_policy == FIFO ) {
                    if ( line->m_alloc_time < valid_timestamp ) {
                        valid_timestamp = line->m_alloc_time;
                        valid_line = index;
                    }
                }
            }
            
        }
    }
    if ( all_reserved ) {
        assert( m_config.m_alloc_policy == ON_MISS );
        return RESERVATION_FAIL; // miss and not enough space in cache to allocate on miss
    }
    if ( invalid_line != (unsigned)-1 ) {
        idx = invalid_line;
        L1_status_invaild=1;   
    } else if ( valid_line != (unsigned)-1) {
        idx = valid_line;
    } else abort(); // if an unreserved block exists, it is either invalid or replaceable 
    cache_block_t *line = &m_lines[idx];
    L1_temp_ishit = line->ishit;
    L1_temp_miss_div = line->block_div_num;
    //printf("probe_miss_div=%d  block_div_num =%d  status =%d\n",L1_temp_miss_div,line->block_div_num,L1_status_invaild);
    if(L1_temp_ishit != -1){
        if(L1_temp_ishit<10)
            L1_ishit_count[L1_temp_ishit]++;
        else
            L1_ishit_count[10]++;
    }
    return MISS;
}
//Zu_Hao:L1 cache probe_div_match
enum cache_request_status tag_array::L2_probe_div_match( new_addr_type addr, unsigned &idx ,int  request_div) const {
    //assert( m_config.m_write_policy == READ_ONLY );
    L2_temp_ishit= -1;
    L2_temp_miss_div=-1;
    L2_status_invaild=0;
    unsigned set_index = m_config.set_index(addr);
    new_addr_type tag = m_config.tag(addr);        

    unsigned invalid_line = (unsigned)-1;
    unsigned valid_line = (unsigned)-1;
    unsigned valid_timestamp = (unsigned)-1;
    bool all_reserved = true;
    // check for hit or pending hit
    for (unsigned way=0; way<m_config.m_assoc; way++) {
        unsigned index = set_index*m_config.m_assoc+way;
        cache_block_t *line = &m_lines[index];
        if (line->m_tag == tag) {  
            if ( line->m_status == RESERVED ) {
                idx = index;
                return HIT_RESERVED;
            } else if ( line->m_status == VALID ) {
                idx = index;
                if ( (request_div>-1) && (request_div<5))
                {
                    if(line->block_div_num != request_div)
                    {
                        L2_warp_div_match[request_div][line->block_div_num]++;
                    }
                    else
                        L2_warp_div_match[request_div][request_div]++;
                }
                L1_temp_div = line->block_div_num;
                line->ishit++;
                return HIT;
            } else if ( line->m_status == MODIFIED ) {
                idx = index;
                if ((request_div > -1) && (request_div < 5))
                {
                    if(line->block_div_num != request_div)
                        L2_warp_div_match[request_div][line->block_div_num]++;
                    else
                        L2_warp_div_match[request_div][request_div]++;
                }
                L1_temp_div = line->block_div_num;
                line->ishit++;
                return HIT;
            } else {
                assert( line->m_status == INVALID );
            }
        }
        if (line->m_status != RESERVED) { 
            all_reserved = false;
            if (line->m_status == INVALID) { //如果有invalid就先選
                invalid_line = index;
            } else {                        //沒有invalid就用LRU  可能是MODIFY或是vaild
                if ( m_config.m_replacement_policy == LRU ) {
                    if ( line->m_last_access_time < valid_timestamp) {
                        valid_timestamp = line->m_last_access_time;
                        valid_line = index;
                    }
                } else if ( m_config.m_replacement_policy == FIFO ) {
                    if ( line->m_alloc_time < valid_timestamp ) {
                        valid_timestamp = line->m_alloc_time;
                        valid_line = index;
                    }
                }
            }
        }
    }
    if ( all_reserved ) {
        assert( m_config.m_alloc_policy == ON_MISS );
        return RESERVATION_FAIL; // miss and not enough space in cache to allocate on miss
    }
    if ( invalid_line != (unsigned)-1 ) {
        idx = invalid_line;
        L2_status_invaild=1;   
    } else if ( valid_line != (unsigned)-1) {
        idx = valid_line;
    } else abort(); // if an unreserved block exists, it is either invalid or replaceable 
    cache_block_t *line = &m_lines[idx];
    L2_temp_ishit = line->ishit;
    L2_temp_miss_div = line->block_div_num;
    //printf("probe_miss_div=%d  block_div_num =%d  status =%d\n",L1_temp_miss_div,line->block_div_num,L1_status_invaild);
    if(L2_temp_ishit != -1){
        if(L2_temp_ishit<10)
            L2_ishit_count[L2_temp_ishit]++;
        else
            L2_ishit_count[10]++;
    }
    return MISS;
}
void victim_cache::keep_cache_line(cache_block_t *Save_Line , cache_block_t *Ass_Line )
{
    Save_Line->block_div_num =  Ass_Line->block_div_num;
    Save_Line->ishit = Ass_Line->ishit;
    Save_Line->m_tag = Ass_Line->m_tag;
    Save_Line->m_block_addr = Ass_Line->m_block_addr;
    Save_Line->m_alloc_time = Ass_Line->m_alloc_time;
    Save_Line->m_last_access_time= Ass_Line->m_last_access_time;
    Save_Line->m_fill_time= Ass_Line->m_fill_time;
    Save_Line->m_status= Ass_Line->m_status;
}
void victim_cache::Line_Swap(unsigned &L1D_idx , unsigned &VC_idx )
{
    cache_block_t temp_line ;
    cache_block_t *L1D_line =new cache_block_t();
    cache_block_t *VC_line = &m_lines[VC_idx];
    keep_cache_line(&temp_line,L1D_line);
    keep_cache_line(L1D_line,VC_line);;
    keep_cache_line(VC_line,&temp_line); 
    
}
int victim_cache::NegTim_count(){
    int NegTim_count =0;
    for (unsigned way=0; way<32; way++) {
        cache_block_t *line = &m_lines[way];
        if(line->m_last_access_time <0)
            NegTim_count++;
    }
    return NegTim_count;
}
enum cache_request_status victim_cache::VC_probe(new_addr_type addr ,unsigned &idx) {
    //printf("VC_probe IN!!\n");
    L1VC_access++;
    int way_count=0;
    new_addr_type tag = m_config.tag(addr);
    unsigned invalid_line = (unsigned)-1;
    unsigned valid_line = (unsigned)-1;
    unsigned valid_timestamp = (unsigned)-1;
    bool all_reserved = true;
    // check for hit or pending hit
    for (unsigned way=0; way<32; way++) {
        way_count++;
        unsigned index = way;//fully associal的作法
        cache_block_t *line = &m_lines[index];
        //printf("probe_line_m_alloc_time = %d  \n",line->m_alloc_time );
        if (line->m_tag == tag) {  //如果HIT的話，line_status只會有valid跟modified
            if ( line->m_status == VALID ) {//全部的cache line應該都會是VALID
                idx = index;
                L1VC_hit++;
                return HIT;
            } else if ( line->m_status == MODIFIED ) {
                idx = index;
                L1VC_hit++;
                return HIT;
            } else {
                assert( line->m_status == INVALID );
            }
            //printf("VC_probe HIT!!\n");
        }
        if (line->m_status != RESERVED) {
            all_reserved = false;
            if (line->m_status == INVALID) {
                invalid_line = index;
            } else { //只有valid 跟 modify 
                // valid line : keep track of most appropriate replacement candidate
                if ( line->m_last_access_time < valid_timestamp ) {
                    valid_timestamp = line->m_last_access_time;
                    valid_line = index;
                }
                
            }
        }
    }
    //printf("way_num=%d  \n",way_count);
    if ( all_reserved ) {
       //printf("VC_probe RF!!\n");
        L1VC_RF++;
        return RESERVATION_FAIL; // miss and not enough space in cache to allocate on miss
    }
    if ( invalid_line != (unsigned)-1 ) {
        idx = invalid_line;
    } else if ( valid_line != (unsigned)-1) {
        idx = valid_line;
    } else {
        //printf("no index!");
        abort(); // if an unreserved block exists, it is either invalid or replaceable
         
    }
   // printf("VC_probe MISS!! , index=%d\n",idx);
    L1VC_miss++;
    //printf("VC_ACCESS = %d VC_HIT =%d VC_MISS =%d L1VC_RF= %d\n",L1VC_access,L1VC_hit,L1VC_miss,L1VC_RF);
    return MISS;
}

enum cache_request_status tag_array::probe( new_addr_type addr, unsigned &idx ) const {
    //assert( m_config.m_write_policy == READ1_ONLY );
    unsigned set_index = m_config.set_index(addr);
    new_addr_type tag = m_config.tag(addr);

    unsigned invalid_line = (unsigned)-1;
    unsigned valid_line = (unsigned)-1;
    unsigned valid_timestamp = (unsigned)-1;

    bool all_reserved = true;

    // check for hit or pending hit
    for (unsigned way=0; way<m_config.m_assoc; way++) {
        unsigned index = set_index*m_config.m_assoc+way;
        cache_block_t *line = &m_lines[index];
        if (line->m_tag == tag) {
            if ( line->m_status == RESERVED ) {
                idx = index;
                return HIT_RESERVED;
            } else if ( line->m_status == VALID ) {
                idx = index;
                return HIT;
            } else if ( line->m_status == MODIFIED ) {
                idx = index;
                return HIT;
            } else {
                assert( line->m_status == INVALID );
            }
        }
        if (line->m_status != RESERVED) {
            all_reserved = false;
            if (line->m_status == INVALID) {
                invalid_line = index;
            } else {
                // valid line : keep track of most appropriate replacement candidate
                if ( m_config.m_replacement_policy == LRU ) {
                    if ( line->m_last_access_time < valid_timestamp ) {
                        valid_timestamp = line->m_last_access_time;
                        valid_line = index;
                    }
                } else if ( m_config.m_replacement_policy == FIFO ) {
                    if ( line->m_alloc_time < valid_timestamp ) {
                        valid_timestamp = line->m_alloc_time;
                        valid_line = index;
                    }
                }
            }
        }
    }
    if ( all_reserved ) {
        assert( m_config.m_alloc_policy == ON_MISS ); 
        return RESERVATION_FAIL; // miss and not enough space in cache to allocate on miss
    }

    if ( invalid_line != (unsigned)-1 ) {
        idx = invalid_line;
    } else if ( valid_line != (unsigned)-1) {
        idx = valid_line;
    } else abort(); // if an unreserved block exists, it is either invalid or replaceable 
    return MISS;
}

enum cache_request_status tag_array::access( new_addr_type addr, unsigned time, unsigned &idx , unsigned request_div)
{
    bool wb=false;
    cache_block_t evicted;
    enum cache_request_status result = access(addr,time,idx,wb,evicted,request_div);
    assert(!wb);
    return result;
}

enum cache_request_status tag_array::access( new_addr_type addr, unsigned time, unsigned &idx, bool &wb, cache_block_t &evicted,unsigned request_div ) 
{
    m_access++;
    shader_cache_access_log(m_core_id, m_type_id, 0); // log accesses to cache
    enum cache_request_status status = probe(addr,idx);
    switch (status) {
    case HIT_RESERVED: 
        m_pending_hit++;
    case HIT: 
        m_lines[idx].m_last_access_time=time; 
        break;
    case MISS:
        m_miss++;
        shader_cache_access_log(m_core_id, m_type_id, 1); // log cache misses
        if ( m_config.m_alloc_policy == ON_MISS ) {
            if( m_lines[idx].m_status == MODIFIED ) {
                wb = true;
                evicted = m_lines[idx];
                wb_div = m_lines[idx].block_div_num;
            }
            m_lines[idx].allocate( m_config.tag(addr), m_config.block_addr(addr), time );
            m_lines[idx].set_block_div(request_div);
        }
        break;
    case RESERVATION_FAIL:
        m_res_fail++;
        shader_cache_access_log(m_core_id, m_type_id, 1); // log cache misses
        break;
    default:
        fprintf( stderr, "tag_array::access - Error: Unknown"
            "cache_request_status %d\n", status );
        abort();
    }
    return status;
}

void tag_array::fill( new_addr_type addr, unsigned time )
{
    assert( m_config.m_alloc_policy == ON_FILL );
    unsigned idx;
    enum cache_request_status status = probe(addr,idx);
    assert(status==MISS); // MSHR should have prevented redundant memory request
    m_lines[idx].allocate( m_config.tag(addr), m_config.block_addr(addr), time );
    m_lines[idx].fill(time);
}

void tag_array::fill( unsigned index, unsigned time ) 
{
    assert( m_config.m_alloc_policy == ON_MISS );
    m_lines[index].fill(time);
}

void tag_array::flush() 
{
    for (unsigned i=0; i < m_config.get_num_lines(); i++)
        m_lines[i].m_status = INVALID;
}

float tag_array::windowed_miss_rate( ) const
{
    unsigned n_access    = m_access - m_prev_snapshot_access;
    unsigned n_miss      = m_miss - m_prev_snapshot_miss;
    // unsigned n_pending_hit = m_pending_hit - m_prev_snapshot_pending_hit;

    float missrate = 0.0f;
    if (n_access != 0)
        missrate = (float) n_miss / n_access;
    return missrate;
}

void tag_array::new_window()
{
    m_prev_snapshot_access = m_access;
    m_prev_snapshot_miss = m_miss;
    m_prev_snapshot_pending_hit = m_pending_hit;
}

void tag_array::print( FILE *stream, unsigned &total_access, unsigned &total_misses ) const
{
    m_config.print(stream);
    fprintf( stream, "\t\tAccess = %d, Miss = %d (%.3g), PendingHit = %d (%.3g)\n", 
             m_access, m_miss, (float) m_miss / m_access, 
             m_pending_hit, (float) m_pending_hit / m_access);
    total_misses+=m_miss;
    total_access+=m_access;
}

void tag_array::get_stats(unsigned &total_access, unsigned &total_misses, unsigned &total_hit_res, unsigned &total_res_fail) const{
    // Update statistics from the tag array
    total_access    = m_access;
    total_misses    = m_miss;
    total_hit_res   = m_pending_hit;
    total_res_fail  = m_res_fail;
}


bool was_write_sent( const std::list<cache_event> &events )
{
    for( std::list<cache_event>::const_iterator e=events.begin(); e!=events.end(); e++ ) {
        if( *e == WRITE_REQUEST_SENT ) 
            return true;
    }
    return false;
}

bool was_writeback_sent( const std::list<cache_event> &events )
{
    for( std::list<cache_event>::const_iterator e=events.begin(); e!=events.end(); e++ ) {
        if( *e == WRITE_BACK_REQUEST_SENT ) 
            return true;
    }
    return false;
}

bool was_read_sent( const std::list<cache_event> &events )
{
    for( std::list<cache_event>::const_iterator e=events.begin(); e!=events.end(); e++ ) {
        if( *e == READ_REQUEST_SENT ) 
            return true;
    }
    return false;
}
/****************************************************************** MSHR ******************************************************************/

/// Checks if there is a pending request to the lower memory level already
bool mshr_table::probe( new_addr_type block_addr ) const{
    table::const_iterator a = m_data.find(block_addr);
    return a != m_data.end();
}

/// Checks if there is space for tracking a new memory access
bool mshr_table::full( new_addr_type block_addr ) const{
    table::const_iterator i=m_data.find(block_addr);
    if ( i != m_data.end() )
        return i->second.m_list.size() >= m_max_merged;
    else
        return m_data.size() >= m_num_entries;
}

/// Add or merge this access
void mshr_table::add( new_addr_type block_addr, mem_fetch *mf ){
    m_data[block_addr].m_list.push_back(mf);
    assert( m_data.size() <= m_num_entries );
    assert( m_data[block_addr].m_list.size() <= m_max_merged );
    // indicate that this MSHR entry contains an atomic operation
    if ( mf->isatomic() ) {
        m_data[block_addr].m_has_atomic = true;
    }
}

/// Accept a new cache fill response: mark entry ready for processing
void mshr_table::mark_ready( new_addr_type block_addr, bool &has_atomic ){
    assert( !busy() );
    table::iterator a = m_data.find(block_addr);
    assert( a != m_data.end() ); // don't remove same request twice
    m_current_response.push_back( block_addr );
    has_atomic = a->second.m_has_atomic;
    assert( m_current_response.size() <= m_data.size() );
}

/// Returns next ready access
mem_fetch *mshr_table::next_access(){
    assert( access_ready() );
    new_addr_type block_addr = m_current_response.front();
    assert( !m_data[block_addr].m_list.empty() );
    mem_fetch *result = m_data[block_addr].m_list.front();
    m_data[block_addr].m_list.pop_front();
    if ( m_data[block_addr].m_list.empty() ) {
        // release entry
        m_data.erase(block_addr);
        m_current_response.pop_front();
    }
    return result;
}

void mshr_table::display( FILE *fp ) const{
    fprintf(fp,"MSHR contents\n");
    for ( table::const_iterator e=m_data.begin(); e!=m_data.end(); ++e ) {
        unsigned block_addr = e->first;
        fprintf(fp,"MSHR: tag=0x%06x, atomic=%d %zu entries : ", block_addr, e->second.m_has_atomic, e->second.m_list.size());
        if ( !e->second.m_list.empty() ) {
            mem_fetch *mf = e->second.m_list.front();
            fprintf(fp,"%p :",mf);
            mf->print(fp);
        } else {
            fprintf(fp," no memory requests???\n");
        }
    }
}
/***************************************************************** Caches *****************************************************************/
cache_stats::cache_stats(){
    m_stats.resize(NUM_MEM_ACCESS_TYPE);
    for(unsigned i=0; i<NUM_MEM_ACCESS_TYPE; ++i){
        m_stats[i].resize(NUM_CACHE_REQUEST_STATUS, 0);
    }
    m_cache_port_available_cycles = 0; 
    m_cache_data_port_busy_cycles = 0; 
    m_cache_fill_port_busy_cycles = 0; 
}

void cache_stats::clear(){
    ///
    /// Zero out all current cache statistics
    ///
    for(unsigned i=0; i<NUM_MEM_ACCESS_TYPE; ++i){
        std::fill(m_stats[i].begin(), m_stats[i].end(), 0);
    }
    m_cache_port_available_cycles = 0; 
    m_cache_data_port_busy_cycles = 0; 
    m_cache_fill_port_busy_cycles = 0; 
}

void cache_stats::inc_stats(int access_type, int access_outcome){
    ///
    /// Increment the stat corresponding to (access_type, access_outcome) by 1.
    ///
    if(!check_valid(access_type, access_outcome))
        assert(0 && "Unknown cache access type or access outcome");
    m_stats[access_type][access_outcome]++;
}

enum cache_request_status cache_stats::select_stats_status(enum cache_request_status probe, enum cache_request_status access) const {
    ///
    /// This function selects how the cache access outcome should be counted. HIT_RESERVED is considered as a MISS
    /// in the cores, however, it should be counted as a HIT_RESERVED in the caches.
    ///
    if(probe == HIT_RESERVED && access != RESERVATION_FAIL)
        return probe;
    else
        return access;
}

unsigned &cache_stats::operator()(int access_type, int access_outcome){
    ///
    /// Simple method to read/modify the stat corresponding to (access_type, access_outcome)
    /// Used overloaded () to avoid the need for separate read/write member functions
    ///
    if(!check_valid(access_type, access_outcome))
        assert(0 && "Unknown cache access type or access outcome");

    return m_stats[access_type][access_outcome];
}

unsigned cache_stats::operator()(int access_type, int access_outcome) const{
    ///
    /// Const accessor into m_stats.
    ///
    if(!check_valid(access_type, access_outcome))
        assert(0 && "Unknown cache access type or access outcome");

    return m_stats[access_type][access_outcome];
}

cache_stats cache_stats::operator+(const cache_stats &cs){
    ///
    /// Overloaded + operator to allow for simple stat accumulation
    ///
    cache_stats ret;
    for(unsigned type=0; type<NUM_MEM_ACCESS_TYPE; ++type){
        for(unsigned status=0; status<NUM_CACHE_REQUEST_STATUS; ++status){
            ret(type, status) = m_stats[type][status] + cs(type, status);
        }
    }
    ret.m_cache_port_available_cycles = m_cache_port_available_cycles + cs.m_cache_port_available_cycles; 
    ret.m_cache_data_port_busy_cycles = m_cache_data_port_busy_cycles + cs.m_cache_data_port_busy_cycles; 
    ret.m_cache_fill_port_busy_cycles = m_cache_fill_port_busy_cycles + cs.m_cache_fill_port_busy_cycles; 
    return ret;
}

cache_stats &cache_stats::operator+=(const cache_stats &cs){
    ///
    /// Overloaded += operator to allow for simple stat accumulation
    ///
    for(unsigned type=0; type<NUM_MEM_ACCESS_TYPE; ++type){
        for(unsigned status=0; status<NUM_CACHE_REQUEST_STATUS; ++status){
            m_stats[type][status] += cs(type, status);
        }
    }
    m_cache_port_available_cycles += cs.m_cache_port_available_cycles; 
    m_cache_data_port_busy_cycles += cs.m_cache_data_port_busy_cycles; 
    m_cache_fill_port_busy_cycles += cs.m_cache_fill_port_busy_cycles; 
    return *this;
}

void cache_stats::print_stats(FILE *fout, const char *cache_name) const{
    ///
    /// Print out each non-zero cache statistic for every memory access type and status
    /// "cache_name" defaults to "Cache_stats" when no argument is provided, otherwise
    /// the provided name is used.
    /// The printed format is "<cache_name>[<request_type>][<request_status>] = <stat_value>"
    ///
    std::string m_cache_name = cache_name;
    for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
        for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status) {
            if(m_stats[type][status] > 0){
                fprintf(fout, "\t%s[%s][%s] = %u\n",
                    m_cache_name.c_str(),
                    mem_access_type_str((enum mem_access_type)type),
                    cache_request_status_str((enum cache_request_status)status),
                    m_stats[type][status]);
            }
        }
    }
}

void cache_sub_stats::print_port_stats(FILE *fout, const char *cache_name) const
{
    float data_port_util = 0.0f; 
    if (port_available_cycles > 0) {
        data_port_util = (float) data_port_busy_cycles / port_available_cycles; 
    }
    fprintf(fout, "%s_data_port_util = %.3f\n", cache_name, data_port_util); 
    float fill_port_util = 0.0f; 
    if (port_available_cycles > 0) {
        fill_port_util = (float) fill_port_busy_cycles / port_available_cycles; 
    }
    fprintf(fout, "%s_fill_port_util = %.3f\n", cache_name, fill_port_util); 
}

unsigned cache_stats::get_stats(enum mem_access_type *access_type, unsigned num_access_type, enum cache_request_status *access_status, unsigned num_access_status) const{
    ///
    /// Returns a sum of the stats corresponding to each "access_type" and "access_status" pair.
    /// "access_type" is an array of "num_access_type" mem_access_types.
    /// "access_status" is an array of "num_access_status" cache_request_statuses.
    ///
    unsigned total=0;
    for(unsigned type =0; type < num_access_type; ++type){
        for(unsigned status=0; status < num_access_status; ++status){
            if(!check_valid((int)access_type[type], (int)access_status[status]))
                assert(0 && "Unknown cache access type or access outcome");
            total += m_stats[access_type[type]][access_status[status]];
        }
    }
    return total;
}
void cache_stats::get_sub_stats(struct cache_sub_stats &css) const{
    ///
    /// Overwrites "css" with the appropriate statistics from this cache.
    ///
    struct cache_sub_stats t_css;
    t_css.clear();

    for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
        for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status) {
            if(status == HIT || status == MISS || status == HIT_RESERVED)
                t_css.accesses += m_stats[type][status];
            if(status == MISS)
                t_css.misses += m_stats[type][status];

            if(status == HIT_RESERVED)
                t_css.pending_hits += m_stats[type][status];

            if(status == RESERVATION_FAIL)
                t_css.res_fails += m_stats[type][status];
        }
    }

    t_css.port_available_cycles = m_cache_port_available_cycles; 
    t_css.data_port_busy_cycles = m_cache_data_port_busy_cycles; 
    t_css.fill_port_busy_cycles = m_cache_fill_port_busy_cycles; 

    css = t_css;
}

bool cache_stats::check_valid(int type, int status) const{
    ///
    /// Verify a valid access_type/access_status
    ///
    if((type >= 0) && (type < NUM_MEM_ACCESS_TYPE) && (status >= 0) && (status < NUM_CACHE_REQUEST_STATUS))
        return true;
    else
        return false;
}

void cache_stats::sample_cache_port_utility(bool data_port_busy, bool fill_port_busy) 
{
    m_cache_port_available_cycles += 1; 
    if (data_port_busy) {
        m_cache_data_port_busy_cycles += 1; 
    } 
    if (fill_port_busy) {
        m_cache_fill_port_busy_cycles += 1; 
    } 
}

baseline_cache::bandwidth_management::bandwidth_management(cache_config &config) 
: m_config(config)
{
    m_data_port_occupied_cycles = 0; 
    m_fill_port_occupied_cycles = 0; 
}

/// use the data port based on the outcome and events generated by the mem_fetch request 
void baseline_cache::bandwidth_management::use_data_port(mem_fetch *mf, enum cache_request_status outcome, const std::list<cache_event> &events)
{
    unsigned data_size = mf->get_data_size(); 
    unsigned port_width = m_config.m_data_port_width; 
    switch (outcome) {
    case HIT: {
        unsigned data_cycles = data_size / port_width + ((data_size % port_width > 0)? 1 : 0); 
        m_data_port_occupied_cycles += data_cycles; 
        } break; 
    case HIT_RESERVED: 
    case MISS: {
        // the data array is accessed to read out the entire line for write-back 
        if (was_writeback_sent(events)) {
            unsigned data_cycles = m_config.m_line_sz / port_width; 
            m_data_port_occupied_cycles += data_cycles; 
        }
        } break; 
    case RESERVATION_FAIL: 
        // Does not consume any port bandwidth 
        break; 
    default: 
        assert(0); 
        break; 
    } 
}

/// use the fill port 
void baseline_cache::bandwidth_management::use_fill_port(mem_fetch *mf)
{
    // assume filling the entire line with the returned request 
    unsigned fill_cycles = m_config.m_line_sz / m_config.m_data_port_width; 
    m_fill_port_occupied_cycles += fill_cycles; 
}

/// called every cache cycle to free up the ports 
void baseline_cache::bandwidth_management::replenish_port_bandwidth()
{
    if (m_data_port_occupied_cycles > 0) {
        m_data_port_occupied_cycles -= 1; 
    }
    assert(m_data_port_occupied_cycles >= 0); 

    if (m_fill_port_occupied_cycles > 0) {
        m_fill_port_occupied_cycles -= 1; 
    }
    assert(m_fill_port_occupied_cycles >= 0); 
}

/// query for data port availability 
bool baseline_cache::bandwidth_management::data_port_free() const
{
    return (m_data_port_occupied_cycles == 0); 
}

/// query for fill port availability 
bool baseline_cache::bandwidth_management::fill_port_free() const
{
    return (m_fill_port_occupied_cycles == 0); 
}

/// Sends next request to lower level of memory
void baseline_cache::cycle(){
    if ( !m_miss_queue.empty() ) {
        mem_fetch *mf = m_miss_queue.front();
        if ( !m_memport->full(mf->size(),mf->get_is_write()) ) {
            m_miss_queue.pop_front();
            m_memport->push(mf);
        }
    }
    bool data_port_busy = !m_bandwidth_management.data_port_free(); 
    bool fill_port_busy = !m_bandwidth_management.fill_port_free(); 
    m_stats.sample_cache_port_utility(data_port_busy, fill_port_busy); 
    m_bandwidth_management.replenish_port_bandwidth(); 
}

/// Interface for response from lower memory level (model bandwidth restictions in caller)
void baseline_cache::fill(mem_fetch *mf, unsigned time){
    extra_mf_fields_lookup::iterator e = m_extra_mf_fields.find(mf);
    assert( e != m_extra_mf_fields.end() );
    assert( e->second.m_valid );
    mf->set_data_size( e->second.m_data_size );
    if ( m_config.m_alloc_policy == ON_MISS )
        m_tag_array->fill(e->second.m_cache_index,time);
    else if ( m_config.m_alloc_policy == ON_FILL )
        m_tag_array->fill(e->second.m_block_addr,time);
    else abort();
    bool has_atomic = false;
    m_mshrs.mark_ready(e->second.m_block_addr, has_atomic);
    if (has_atomic) {
        assert(m_config.m_alloc_policy == ON_MISS);
        cache_block_t &block = m_tag_array->get_block(e->second.m_cache_index);
        block.m_status = MODIFIED; // mark line as dirty for atomic operation
    }
    m_extra_mf_fields.erase(mf);
    m_bandwidth_management.use_fill_port(mf); 
}

/// Checks if mf is waiting to be filled by lower memory level
bool baseline_cache::waiting_for_fill( mem_fetch *mf ){
    extra_mf_fields_lookup::iterator e = m_extra_mf_fields.find(mf);
    return e != m_extra_mf_fields.end();
}

void baseline_cache::print(FILE *fp, unsigned &accesses, unsigned &misses) const{
    fprintf( fp, "Cache %s:\t", m_name.c_str() );
    m_tag_array->print(fp,accesses,misses);
}

void baseline_cache::display_state( FILE *fp ) const{
    fprintf(fp,"Cache %s:\n", m_name.c_str() );
    m_mshrs.display(fp);
    fprintf(fp,"\n");
}

/// Read miss handler without writeback
void baseline_cache::send_read_request(new_addr_type addr, new_addr_type block_addr, unsigned cache_index, mem_fetch *mf,
        unsigned time, bool &do_miss, std::list<cache_event> &events, bool read_only, bool wa){

    bool wb=false;
    cache_block_t e;
    send_read_request(addr, block_addr, cache_index, mf, time, do_miss, wb, e, events, read_only, wa);
}

/// Read miss handler. Check MSHR hit or MSHR available
void baseline_cache::send_read_request(new_addr_type addr, new_addr_type block_addr, unsigned cache_index, mem_fetch *mf,
        unsigned time, bool &do_miss, bool &wb, cache_block_t &evicted, std::list<cache_event> &events, bool read_only, bool wa){

    bool mshr_hit = m_mshrs.probe(block_addr);
    bool mshr_avail = !m_mshrs.full(block_addr);
    if ( mshr_hit && mshr_avail ) {
        if(read_only)
            m_tag_array->access(block_addr,time,cache_index,mf->mf_div);
        else
            m_tag_array->access(block_addr,time,cache_index,wb,evicted,mf->mf_div);

        m_mshrs.add(block_addr,mf);
        do_miss = true;
    } else if ( !mshr_hit && mshr_avail && (m_miss_queue.size() < m_config.m_miss_queue_size) ) {
        if(read_only)
            m_tag_array->access(block_addr,time,cache_index,mf->mf_div);
        else
            m_tag_array->access(block_addr,time,cache_index,wb,evicted,mf->mf_div);

        m_mshrs.add(block_addr,mf);
        m_extra_mf_fields[mf] = extra_mf_fields(block_addr,cache_index, mf->get_data_size());
        mf->set_data_size( m_config.get_line_sz() );
        m_miss_queue.push_back(mf);
        mf->set_status(m_miss_queue_status,time);
        if(!wa)
            events.push_back(READ_REQUEST_SENT);
        do_miss = true;
    }
}


/// Sends write request to lower level memory (write or writeback)
void data_cache::send_write_request(mem_fetch *mf, cache_event request, unsigned time, std::list<cache_event> &events){
    events.push_back(request);
    m_miss_queue.push_back(mf);
    mf->set_status(m_miss_queue_status,time);
}


/****** Write-hit functions (Set by config file) ******/

/// Write-back hit: Mark block as modified
cache_request_status data_cache::wr_hit_wb(new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time, std::list<cache_event> &events, enum cache_request_status status ){
    new_addr_type block_addr = m_config.block_addr(addr);
    m_tag_array->access(block_addr,time,cache_index,mf->mf_div); // update LRU state
    cache_block_t &block = m_tag_array->get_block(cache_index);
    block.m_status = MODIFIED;

    return HIT;
}

/// Write-through hit: Directly send request to lower level memory
cache_request_status data_cache::wr_hit_wt(new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time, std::list<cache_event> &events, enum cache_request_status status ){
    if(miss_queue_full(0))
        return RESERVATION_FAIL; // cannot handle request this cycle

    new_addr_type block_addr = m_config.block_addr(addr);
    m_tag_array->access(block_addr,time,cache_index,mf->mf_div); // update LRU state
    cache_block_t &block = m_tag_array->get_block(cache_index);
    block.m_status = MODIFIED;

    // generate a write-through
    send_write_request(mf, WRITE_REQUEST_SENT, time, events);

    return HIT;
}

/// Write-evict hit: Send request to lower level memory and invalidate corresponding block
cache_request_status data_cache::wr_hit_we(new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time, std::list<cache_event> &events, enum cache_request_status status ){
    if(miss_queue_full(0))
        return RESERVATION_FAIL; // cannot handle request this cycle

    // generate a write-through/evict
    cache_block_t &block = m_tag_array->get_block(cache_index);
    send_write_request(mf, WRITE_REQUEST_SENT, time, events);

    // Invalidate block
    block.m_status = INVALID;

    return HIT;
}

/// Global write-evict, local write-back: Useful for private caches
enum cache_request_status data_cache::wr_hit_global_we_local_wb(new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time, std::list<cache_event> &events, enum cache_request_status status ){
    bool evict = (mf->get_access_type() == GLOBAL_ACC_W); // evict a line that hits on global memory write
    if(evict)
        return wr_hit_we(addr, cache_index, mf, time, events, status); // Write-evict
    else
        return wr_hit_wb(addr, cache_index, mf, time, events, status); // Write-back
}

/****** Write-miss functions (Set by config file) ******/

/// Write-allocate miss: Send write request to lower level memory
// and send a read request for the same block
enum cache_request_status
data_cache::wr_miss_wa( new_addr_type addr,
                        unsigned cache_index, mem_fetch *mf,
                        unsigned time, std::list<cache_event> &events,
                        enum cache_request_status status )
{
    new_addr_type block_addr = m_config.block_addr(addr);

    // Write allocate, maximum 3 requests (write miss, read request, write back request)
    // Conservatively ensure the worst-case request can be handled this cycle
    bool mshr_hit = m_mshrs.probe(block_addr);
    bool mshr_avail = !m_mshrs.full(block_addr);
    if(miss_queue_full(2) 
        || (!(mshr_hit && mshr_avail) 
        && !(!mshr_hit && mshr_avail 
        && (m_miss_queue.size() < m_config.m_miss_queue_size))))
        return RESERVATION_FAIL;

    send_write_request(mf, WRITE_REQUEST_SENT, time, events);
    // Tries to send write allocate request, returns true on success and false on failure
    //if(!send_write_allocate(mf, addr, block_addr, cache_index, time, events))
    //    return RESERVATION_FAIL;

    const mem_access_t *ma = new  mem_access_t( m_wr_alloc_type,
                        mf->get_addr(),
                        mf->get_data_size(),
                        false, // Now performing a read
                        mf->get_access_warp_mask(),
                        mf->get_access_byte_mask() );

    mem_fetch *n_mf = new mem_fetch( *ma,
                    NULL,
                    mf->get_ctrl_size(),
                    mf->get_wid(),
                    mf->get_sid(),
                    mf->get_tpc(),
                    mf->get_mem_config());
    //Zu_Hao: 因為write miss 會有write allocate所以在send_read_request時mf所記錄的mf_div是初始值
    n_mf->mf_div=mf->mf_div;
    bool do_miss = false;
    bool wb = false;
    cache_block_t evicted;

    // Send read request resulting from write miss
    send_read_request(addr, block_addr, cache_index, n_mf, time, do_miss, wb,
        evicted, events, false, true);

    if( do_miss ){
        // If evicted block is modified and not a write-through
        // (already modified lower level)
        if( wb && (m_config.m_write_policy != WRITE_THROUGH) ) { 
            mem_fetch *wb = m_memfetch_creator->alloc(evicted.m_block_addr,
                m_wrbk_type,m_config.get_line_sz(),true);
            m_miss_queue.push_back(wb);
            wb->set_status(m_miss_queue_status,time);
        }
        return MISS;
    }

    return RESERVATION_FAIL;
}

/// No write-allocate miss: Simply send write request to lower level memory
enum cache_request_status
data_cache::wr_miss_no_wa( new_addr_type addr,
                           unsigned cache_index,
                           mem_fetch *mf,
                           unsigned time,
                           std::list<cache_event> &events,
                           enum cache_request_status status )
{
    if(miss_queue_full(0))
        return RESERVATION_FAIL; // cannot handle request this cycle

    // on miss, generate write through (no write buffering -- too many threads for that)
    send_write_request(mf, WRITE_REQUEST_SENT, time, events);

    return MISS;
}

/****** Read hit functions (Set by config file) ******/

/// Baseline read hit: Update LRU status of block.
// Special case for atomic instructions -> Mark block as modified
enum cache_request_status
data_cache::rd_hit_base( new_addr_type addr,
                         unsigned cache_index,
                         mem_fetch *mf,
                         unsigned time,
                         std::list<cache_event> &events,
                         enum cache_request_status status )
{
    new_addr_type block_addr = m_config.block_addr(addr);
    m_tag_array->access(block_addr,time,cache_index,mf->mf_div);
    // Atomics treated as global read/write requests - Perform read, mark line as
    // MODIFIED
    if(mf->isatomic()){ 
        assert(mf->get_access_type() == GLOBAL_ACC_R);
        cache_block_t &block = m_tag_array->get_block(cache_index);
        block.m_status = MODIFIED;  // mark line as dirty
    }
    return HIT;
}

/****** Read miss functions (Set by config file) ******/

/// Baseline read miss: Send read request to lower level memory,
// perform write-back as necessary
enum cache_request_status
data_cache::rd_miss_base( new_addr_type addr,
                          unsigned cache_index,
                          mem_fetch *mf,
                          unsigned time,
                          std::list<cache_event> &events,
                          enum cache_request_status status ){
    if(miss_queue_full(1))
        // cannot handle request this cycle
        // (might need to generate two requests)
        return RESERVATION_FAIL; 

    new_addr_type block_addr = m_config.block_addr(addr);
    bool do_miss = false;
    bool wb = false;
    cache_block_t evicted;
    send_read_request( addr,
                       block_addr,
                       cache_index,
                       mf, time, do_miss, wb, evicted, events, false, false);

    if( do_miss ){
        // If evicted block is modified and not a write-through
        // (already modified lower level)
        if(wb && (m_config.m_write_policy != WRITE_THROUGH) ){ 
            mem_fetch *wb = m_memfetch_creator->alloc(evicted.m_block_addr,
                m_wrbk_type,m_config.get_line_sz(),true);
            //Zu_Hao: 因為read miss時，原本的cache line要寫回L2，但是不是請求，所以該筆資料的div值要自己改變。
            wb->mf_div = wb_div; 
            //已經把read miss 跟指令的 div都改好了，但是L1跟ori的都還沒改完
            send_write_request(wb, WRITE_BACK_REQUEST_SENT, time, events);
        }
        return MISS;
    }
    return RESERVATION_FAIL;
}

/// Access cache for read_only_cache: returns RESERVATION_FAIL if
// request could not be accepted (for any reason)
enum cache_request_status
read_only_cache::access( new_addr_type addr,
                         mem_fetch *mf,
                         unsigned time,
                         std::list<cache_event> &events )
{
    assert( mf->get_data_size() <= m_config.get_line_sz());
    assert(m_config.m_write_policy == READ_ONLY);
    assert(!mf->get_is_write());
    new_addr_type block_addr = m_config.block_addr(addr);
    unsigned cache_index = (unsigned)-1;
    enum cache_request_status status = m_tag_array->probe(block_addr,cache_index);
    enum cache_request_status cache_status = RESERVATION_FAIL;

    if ( status == HIT ) {
        cache_status = m_tag_array->access(block_addr,time,cache_index,mf->mf_div); // update LRU state
    }else if ( status != RESERVATION_FAIL ) {
        if(!miss_queue_full(0)){
            bool do_miss=false;
            send_read_request(addr, block_addr, cache_index, mf, time, do_miss, events, true, false);
            if(do_miss)
                cache_status = MISS;
            else
                cache_status = RESERVATION_FAIL;
        }else{
            cache_status = RESERVATION_FAIL;
        }
    }

    m_stats.inc_stats(mf->get_access_type(), m_stats.select_stats_status(status, cache_status));
    return cache_status;
}

//! A general function that takes the result of a tag_array probe
//  and performs the correspding functions based on the cache configuration
//  The access fucntion calls this function
enum cache_request_status
data_cache::process_tag_probe( bool wr,
                               enum cache_request_status probe_status,
                               new_addr_type addr,
                               unsigned cache_index,
                               mem_fetch* mf,
                               unsigned time,
                               std::list<cache_event>& events )
{
    // Each function pointer ( m_[rd/wr]_[hit/miss] ) is set in the
    // data_cache constructor to reflect the corresponding cache configuration
    // options. Function pointers were used to avoid many long conditional
    // branches resulting from many cache configuration options.
    cache_request_status access_status = probe_status;
    if(wr){ // Write
        if(probe_status == HIT){
            access_status = (this->*m_wr_hit)( addr,
                                      cache_index,
                                      mf, time, events, probe_status );
        }else if ( probe_status != RESERVATION_FAIL ) {
            access_status = (this->*m_wr_miss)( addr,
                                       cache_index,
                                       mf, time, events, probe_status );
        }
    }else{ // Read
        if(probe_status == HIT){
            access_status = (this->*m_rd_hit)( addr,
                                      cache_index,
                                      mf, time, events, probe_status );
        }else if ( probe_status != RESERVATION_FAIL ) {
            access_status = (this->*m_rd_miss)( addr,
                                       cache_index,
                                       mf, time, events, probe_status );
        }
    }

    m_bandwidth_management.use_data_port(mf, access_status, events); 
    return access_status;
}

// Both the L1 and L2 currently use the same access function.
// Differentiation between the two caches is done through configuration
// of caching policies.
// Both the L1 and L2 override this function to provide a means of
// performing actions specific to each cache when such actions are implemnted.
enum cache_request_status
data_cache::access( new_addr_type addr,
                    mem_fetch *mf,
                    unsigned time,
                    std::list<cache_event> &events )
{

    assert( mf->get_data_size() <= m_config.get_line_sz());
    bool wr = mf->get_is_write();
    new_addr_type block_addr = m_config.block_addr(addr);
    unsigned cache_index = (unsigned)-1;
    enum cache_request_status probe_status
        = m_tag_array->probe( block_addr, cache_index );
    enum cache_request_status access_status
        = process_tag_probe( wr, probe_status, addr, cache_index, mf, time, events );
    m_stats.inc_stats(mf->get_access_type(),
        m_stats.select_stats_status(probe_status, access_status));
    return access_status;
}
//Zu_Hao: L1 victim cache implement
enum cache_request_status
data_cache::access_L1D_VC( new_addr_type addr,
                    mem_fetch *mf,
                    unsigned time,
                    std::list<cache_event> &events ,
                    victim_cache *m_victim_cache,
                    l1_cache *L1D_cache )
{
    /*
    printf("victim_adcddress =%d  L1_cache = %d \n" ,m_victim_cache,L1D_cache);
    for (unsigned way=0; way<32; way++) {
        unsigned index = way;//fully associal的作法
        cache_block_t *line = &(m_victim_cache->m_lines[index]);
        printf("AC_in time = %d \n",line->m_last_access_time);
    }*/
    enum cache_request_status victim_status = RESERVATION_FAIL;
    unsigned VC_index = (unsigned)-1; //VC 命中或是miss的時候得到cache line的idx
    assert( mf->get_data_size() <= m_config.get_line_sz());
    bool wr = mf->get_is_write();
    new_addr_type block_addr = m_config.block_addr(addr);
    unsigned cache_index = (unsigned)-1;
    enum cache_request_status probe_status = m_tag_array->probe_div_match( block_addr, cache_index ,mf->mf_div); 
    ////////////////////victim cache///////////////////////////   
    if (is_use_L1VC && probe_status == MISS ){ //"miss進去VC做事" "RF不做事"  "HIT跟HitP就直接proccess tag"
        victim_status =  m_victim_cache->VC_probe(block_addr,VC_index);  //還沒做好
        //printf("VC_idx =%d  ,L1_temp_miss_div = %d ,L1_temp_ishit =%u ,status= %d victim_status=%d\n",VC_index,L1_temp_miss_div,L1_temp_ishit,L1_status_invaild,victim_status);
        if(victim_status == MISS){    //victim cache miss
            if(L1_temp_miss_div ==1  && L1_temp_ishit > 5 &&L1_status_invaild!=1){
                //printf("%d ,VC=miss goto_line_swap!!!\n",victim_status);
                m_victim_cache->Line_Swap(cache_index,VC_index);
                probe_status = m_tag_array->probe_div_match( block_addr, cache_index ,mf->mf_div);   //確保狀態變成invalid有錯
                //printf("VCmiss,Line_Swap down\n");
            }
            //printf("VC=miss\n");
        }
        else{
            if( victim_status != RESERVATION_FAIL)  //victim hit  如果HIT的時候跟L1D交換可能會把低reuse的換進去，
            {
                m_victim_cache->Line_Swap(cache_index,VC_index);
                m_victim_cache->m_lines[VC_index].m_last_access_time= 0 ;//讓他最早被換出去;
                probe_status = HIT ;
                //printf("VC=HIT, Line_Swap down\n");
            }
        }
    }
    enum cache_request_status access_status
        = process_tag_probe( wr, probe_status, addr, cache_index, mf, time, events );
    //printf("process down!!\n");
    if(probe_status ==2 && access_status !=2)
        if(L1_temp_ishit < 10)
            L1_ishit_count[L1_temp_ishit]--;
        else 
            L1_ishit_count[10]--;
    if(probe_status == 0 && access_status != 0){
        L1_warp_div_match[mf->mf_div-1][L1_temp_div-1]--;
    }
    m_stats.inc_stats(mf->get_access_type(),
        m_stats.select_stats_status(probe_status, access_status));
    /* 
    printf("VC_probe_down!\n");
    for (unsigned way=0; way<32; way++) {
        unsigned index = way;//fully associal的作法
        cache_block_t *line = &(m_victim_cache->m_lines[index]);
        printf("AC_out time = %d \n",line->m_last_access_time);
    }
    */
    return access_status;
}
enum cache_request_status
data_cache::access_L2_VC( new_addr_type addr,
                    mem_fetch *mf,
                    unsigned time,
                    std::list<cache_event> &events ,
                    victim_cache *m_victim_cache,
                    l2_cache *L2_cache )
{
    /*
    printf("victim_adcddress =%d  L1_cache = %d \n" ,m_victim_cache,L2_cache);
    for (unsigned way=0; way<32; way++) {
        unsigned index = way;//fully associal的作法
        cache_block_t *line = &(m_victim_cache->m_lines[index]);
        printf("AC_in time = %d \n",line->m_last_access_time);
    }*/
    enum cache_request_status victim_status = RESERVATION_FAIL;
    unsigned VC_index = (unsigned)-1; //VC 命中或是miss的時候得到cache line的idx
    assert( mf->get_data_size() <= m_config.get_line_sz());
    bool wr = mf->get_is_write();
    new_addr_type block_addr = m_config.block_addr(addr);
    unsigned cache_index = (unsigned)-1;
    enum cache_request_status probe_status = m_tag_array->L2_probe_div_match( block_addr, cache_index ,mf->mf_div); 
    bool is_use_L2VC = 0;   //L2VC開關
    ////////////////////victim cache///////////////////////////   
    if (is_use_L2VC && probe_status == MISS ){ //"miss進去VC做事" "RF不做事"  "HIT跟HitP就直接proccess tag"
        victim_status =  m_victim_cache->VC_probe(block_addr,VC_index);  //還沒做好
        //printf("VC_idx =%d  ,L1_temp_miss_div = %d ,L1_temp_ishit =%u ,status= %d victim_status=%d\n",VC_index,L1_temp_miss_div,L1_temp_ishit,L1_status_invaild,victim_status);
        if(victim_status == MISS){    //victim cache miss
            if(L2_temp_miss_div ==1 && L2_temp_ishit > 5 && L2_status_invaild!=1){
                //printf("%d ,VC=miss goto_line_swap!!!\n",victim_status);
                m_victim_cache->Line_Swap(cache_index,VC_index);
                probe_status = m_tag_array->L2_probe_div_match( block_addr, cache_index ,mf->mf_div);   //確保狀態變成invalid有錯
                //printf("VCmiss,Line_Swap down\n");
            }
            //printf("VC=miss\n");
        }
        else{
            if( victim_status != RESERVATION_FAIL)  //victim hit  如果HIT的時候跟L1D交換可能會把低reuse的換進去，
            {
                m_victim_cache->Line_Swap(cache_index,VC_index);
                m_victim_cache->m_lines[VC_index].m_last_access_time= 0 ;//讓他最早被換出去;
                probe_status = HIT ;
                //printf("VC=HIT, Line_Swap down\n");
            }
        }
    }
    enum cache_request_status access_status
        = process_tag_probe( wr, probe_status, addr, cache_index, mf, time, events );
    //printf("process down!!\n");
    if(probe_status ==2 && access_status !=2)
        if(L2_temp_ishit < 10)
            L2_ishit_count[L2_temp_ishit]--;
        else 
            L2_ishit_count[10]--;
    if(probe_status == 0 && access_status != 0){
        L2_warp_div_match[mf->mf_div-1][L2_temp_div-1]--;
    }
    m_stats.inc_stats(mf->get_access_type(),
        m_stats.select_stats_status(probe_status, access_status));
    /* 
    printf("VC_probe_down!\n");
    for (unsigned way=0; way<32; way++) {
        unsigned index = way;//fully associal的作法
        cache_block_t *line = &(m_victim_cache->m_lines[index]);
        printf("AC_out time = %d \n",line->m_last_access_time);
    }
    */
    return access_status;
}
/// This is meant to model the first level data cache in Fermi.
/// It is write-evict (global) or write-back (local) at the
/// granularity of individual blocks (Set by GPGPU-Sim configuration file)
/// (the policy used in fermi according to the CUDA manual)
enum cache_request_status
l1_cache::access( new_addr_type addr,
                  mem_fetch *mf,
                  unsigned time,
                  std::list<cache_event> &events )
{
//Zu_Hao: use origin L1 data cache or our L1 data cache with victim cache.
    if(!use_bypass_L1cache)
        return data_cache::access( addr, mf, time, events );
    else
        return data_cache::access_L1D_VC( addr, mf, time, events ,&m_victim_cache,this);
}
//Zu_Hao:Printing the distribute of cache hit of request .

void l1_cache::printf_div_match(int *one_to_one,int *one_to_two,int*two_to_one,int *two_to_two)
{ 
    *one_to_one = L1_warp_div_match[0][0];
    *one_to_two = L1_warp_div_match[0][1];
    *two_to_one = L1_warp_div_match[1][0];
    *two_to_two = L1_warp_div_match[1][1];
    //printf("div1_match  =  1to1 = %u , 1to2 = %u \n div2_match = 2to1 = %u ,2to2 = %u \n ",L1_warp_div_match[0][0],L1_warp_div_match[0][1],L1_warp_div_match[1][0],L1_warp_div_match[1][1]);
}   
void l1_cache::printf_ishit(unsigned CL_ishit[])
{
    for(int i = 0 ; i<11 ;i++)
        CL_ishit[i] = L1_ishit_count[i];
    unsigned L2sun_ishit = 0;
    for (int i = 0; i < 11; i++){
        L2sun_ishit+=L2_ishit_count[i];
        printf("L2CL_ishit %d = %d\n",i,L2_ishit_count[i]);
    }
    printf("L2CL_sun = %u\n",L2sun_ishit);;
}
// The l2 cache access function calls the base data_cache access
// implementation.  When the L2 needs to diverge from L1, L2 specific
// changes should be made here.
enum cache_request_status
l2_cache::access( new_addr_type addr,
                  mem_fetch *mf,
                  unsigned time,
                  std::list<cache_event> &events )
{
//Zu_Hao:use origin L2  cache or our l2 cache with victim cache
    if (!use_bypass_L2cache)
        return data_cache::access( addr, mf, time, events );
    else
        return data_cache::access_L2_VC(addr,mf,time,events,&m_victim_l2cache,this);
}

/// Access function for tex_cache
/// return values: RESERVATION_FAIL if request could not be accepted
/// otherwise returns HIT_RESERVED or MISS; NOTE: *never* returns HIT
/// since unlike a normal CPU cache, a "HIT" in texture cache does not
/// mean the data is ready (still need to get through fragment fifo)
enum cache_request_status tex_cache::access( new_addr_type addr, mem_fetch *mf,
    unsigned time, std::list<cache_event> &events )
{
    if ( m_fragment_fifo.full() || m_request_fifo.full() || m_rob.full() )
        return RESERVATION_FAIL;

    assert( mf->get_data_size() <= m_config.get_line_sz());

    // at this point, we will accept the request : access tags and immediately allocate line
    new_addr_type block_addr = m_config.block_addr(addr);
    unsigned cache_index = (unsigned)-1;
    enum cache_request_status status = m_tags.access(block_addr,time,cache_index,mf->mf_div);
    enum cache_request_status cache_status = RESERVATION_FAIL;
    assert( status != RESERVATION_FAIL );
    assert( status != HIT_RESERVED ); // as far as tags are concerned: HIT or MISS
    m_fragment_fifo.push( fragment_entry(mf,cache_index,status==MISS,mf->get_data_size()) );
    if ( status == MISS ) {
        // we need to send a memory request...
        unsigned rob_index = m_rob.push( rob_entry(cache_index, mf, block_addr) );
        m_extra_mf_fields[mf] = extra_mf_fields(rob_index);
        mf->set_data_size(m_config.get_line_sz());
        m_tags.fill(cache_index,time); // mark block as valid
        m_request_fifo.push(mf);
        mf->set_status(m_request_queue_status,time);
        events.push_back(READ_REQUEST_SENT);
        cache_status = MISS;
    } else {
        // the value *will* *be* in the cache already
        cache_status = HIT_RESERVED;
    }
    m_stats.inc_stats(mf->get_access_type(), m_stats.select_stats_status(status, cache_status));
    return cache_status;
}

void tex_cache::cycle(){
    // send next request to lower level of memory
    if ( !m_request_fifo.empty() ) {
        mem_fetch *mf = m_request_fifo.peek();
        if ( !m_memport->full(mf->get_ctrl_size(),false) ) {
            m_request_fifo.pop();
            m_memport->push(mf);
        }
    }
    // read ready lines from cache
    if ( !m_fragment_fifo.empty() && !m_result_fifo.full() ) {
        const fragment_entry &e = m_fragment_fifo.peek();
        if ( e.m_miss ) {
            // check head of reorder buffer to see if data is back from memory
            unsigned rob_index = m_rob.next_pop_index();
            const rob_entry &r = m_rob.peek(rob_index);
            assert( r.m_request == e.m_request );
            assert( r.m_block_addr == m_config.block_addr(e.m_request->get_addr()) );
            if ( r.m_ready ) {
                assert( r.m_index == e.m_cache_index );
                m_cache[r.m_index].m_valid = true;
                m_cache[r.m_index].m_block_addr = r.m_block_addr;
                m_result_fifo.push(e.m_request);
                m_rob.pop();
                m_fragment_fifo.pop();
            }
        } else {
            // hit:
            assert( m_cache[e.m_cache_index].m_valid );
            assert( m_cache[e.m_cache_index].m_block_addr
                == m_config.block_addr(e.m_request->get_addr()) );
            m_result_fifo.push( e.m_request );
            m_fragment_fifo.pop();
        }
    }
}

/// Place returning cache block into reorder buffer
void tex_cache::fill( mem_fetch *mf, unsigned time )
{
    extra_mf_fields_lookup::iterator e = m_extra_mf_fields.find(mf);
    assert( e != m_extra_mf_fields.end() );
    assert( e->second.m_valid );
    assert( !m_rob.empty() );
    mf->set_status(m_rob_status,time);

    unsigned rob_index = e->second.m_rob_index;
    rob_entry &r = m_rob.peek(rob_index);
    assert( !r.m_ready );
    r.m_ready = true;
    r.m_time = time;
    assert( r.m_block_addr == m_config.block_addr(mf->get_addr()) );
}

void tex_cache::display_state( FILE *fp ) const
{
    fprintf(fp,"%s (texture cache) state:\n", m_name.c_str() );
    fprintf(fp,"fragment fifo entries  = %u / %u\n",
        m_fragment_fifo.size(), m_fragment_fifo.capacity() );
    fprintf(fp,"reorder buffer entries = %u / %u\n",
        m_rob.size(), m_rob.capacity() );
    fprintf(fp,"request fifo entries   = %u / %u\n",
        m_request_fifo.size(), m_request_fifo.capacity() );
    if ( !m_rob.empty() )
        fprintf(fp,"reorder buffer contents:\n");
    for ( int n=m_rob.size()-1; n>=0; n-- ) {
        unsigned index = (m_rob.next_pop_index() + n)%m_rob.capacity();
        const rob_entry &r = m_rob.peek(index);
        fprintf(fp, "tex rob[%3d] : %s ",
            index, (r.m_ready?"ready  ":"pending") );
        if ( r.m_ready )
            fprintf(fp,"@%6u", r.m_time );
        else
            fprintf(fp,"       ");
        fprintf(fp,"[idx=%4u]",r.m_index);
        r.m_request->print(fp,false);
    }
    if ( !m_fragment_fifo.empty() ) {
        fprintf(fp,"fragment fifo (oldest) :");
        fragment_entry &f = m_fragment_fifo.peek();
        fprintf(fp,"%s:          ", f.m_miss?"miss":"hit ");
        f.m_request->print(fp,false);
    }
}
/******************************************************************************************************************************************/


