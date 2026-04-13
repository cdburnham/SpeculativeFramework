#ifndef OSDZU3_JSON_H
#define OSDZU3_JSON_H

#include "osdzu3_common.h"

typedef enum {
    OSDZU3_JSON_UNDEFINED = 0,
    OSDZU3_JSON_OBJECT,
    OSDZU3_JSON_ARRAY,
    OSDZU3_JSON_STRING,
    OSDZU3_JSON_PRIMITIVE
} osdzu3_json_type_t;

typedef struct {
    osdzu3_json_type_t type;
    int start;
    int end;
    int size;
    int parent;
} osdzu3_json_token_t;

typedef struct {
    unsigned int pos;
    unsigned int next_token;
    int current_parent;
} osdzu3_json_parser_t;

void osdzu3_json_init(osdzu3_json_parser_t *parser);
int osdzu3_json_tokenize(osdzu3_json_parser_t *parser,
                         const char *json,
                         size_t len,
                         osdzu3_json_token_t *tokens,
                         unsigned int max_tokens);
int osdzu3_json_token_next(const osdzu3_json_token_t *tokens, int count, int index);
bool osdzu3_json_token_equals(const char *json,
                              const osdzu3_json_token_t *token,
                              const char *text);
int osdzu3_json_object_get(const char *json,
                           const osdzu3_json_token_t *tokens,
                           int count,
                           int object_index,
                           const char *key);
int osdzu3_json_array_get(const osdzu3_json_token_t *tokens,
                          int count,
                          int array_index,
                          int element_index);
int osdzu3_json_token_to_string(const char *json,
                                const osdzu3_json_token_t *token,
                                char *out,
                                size_t out_size);
bool osdzu3_json_token_to_u32(const char *json,
                              const osdzu3_json_token_t *token,
                              uint32_t *value);
bool osdzu3_json_token_to_float(const char *json,
                                const osdzu3_json_token_t *token,
                                float *value);
bool osdzu3_json_token_to_bool(const char *json,
                               const osdzu3_json_token_t *token,
                               bool *value);

#endif
