#include "osdzu3_json.h"

#include <ctype.h>
#include <stdlib.h>
#include <string.h>

static osdzu3_json_token_t *osdzu3_json_alloc_token(osdzu3_json_parser_t *parser,
                                                    osdzu3_json_token_t *tokens,
                                                    unsigned int max_tokens) {
    osdzu3_json_token_t *token;

    if (parser->next_token >= max_tokens) {
        return NULL;
    }

    token = &tokens[parser->next_token++];
    token->start = -1;
    token->end = -1;
    token->size = 0;
    token->parent = -1;
    token->type = OSDZU3_JSON_UNDEFINED;
    return token;
}

static void osdzu3_json_fill_token(osdzu3_json_token_t *token,
                                   osdzu3_json_type_t type,
                                   int start,
                                   int end) {
    token->type = type;
    token->start = start;
    token->end = end;
    token->size = 0;
}

void osdzu3_json_init(osdzu3_json_parser_t *parser) {
    parser->pos = 0;
    parser->next_token = 0;
    parser->current_parent = -1;
}

static int osdzu3_json_parse_primitive(osdzu3_json_parser_t *parser,
                                       const char *json,
                                       size_t len,
                                       osdzu3_json_token_t *tokens,
                                       unsigned int max_tokens) {
    int start = (int) parser->pos;
    osdzu3_json_token_t *token;

    while (parser->pos < len) {
        char c = json[parser->pos];
        if (c == '\t' || c == '\r' || c == '\n' || c == ' ' ||
            c == ',' || c == ']' || c == '}') {
            break;
        }
        if ((unsigned char) c < 32U) {
            return -1;
        }
        parser->pos++;
    }

    token = osdzu3_json_alloc_token(parser, tokens, max_tokens);
    if (token == NULL) {
        return -1;
    }
    osdzu3_json_fill_token(token, OSDZU3_JSON_PRIMITIVE, start, (int) parser->pos);
    token->parent = parser->current_parent;
    parser->pos--;
    return 0;
}

static int osdzu3_json_parse_string(osdzu3_json_parser_t *parser,
                                    const char *json,
                                    size_t len,
                                    osdzu3_json_token_t *tokens,
                                    unsigned int max_tokens) {
    int start = (int) parser->pos + 1;
    osdzu3_json_token_t *token;

    parser->pos++;
    while (parser->pos < len) {
        char c = json[parser->pos];
        if (c == '\"') {
            token = osdzu3_json_alloc_token(parser, tokens, max_tokens);
            if (token == NULL) {
                return -1;
            }
            osdzu3_json_fill_token(token, OSDZU3_JSON_STRING, start, (int) parser->pos);
            token->parent = parser->current_parent;
            return 0;
        }
        if (c == '\\') {
            parser->pos++;
            if (parser->pos >= len) {
                return -1;
            }
        }
        parser->pos++;
    }

    return -1;
}

int osdzu3_json_tokenize(osdzu3_json_parser_t *parser,
                         const char *json,
                         size_t len,
                         osdzu3_json_token_t *tokens,
                         unsigned int max_tokens) {
    unsigned int i;

    for (i = 0; i < len; i++) {
        char c = json[i];
        parser->pos = i;
        switch (c) {
            case '{':
            case '[': {
                osdzu3_json_type_t type = (c == '{') ? OSDZU3_JSON_OBJECT : OSDZU3_JSON_ARRAY;
                osdzu3_json_token_t *token = osdzu3_json_alloc_token(parser, tokens, max_tokens);
                if (token == NULL) {
                    return -1;
                }
                osdzu3_json_fill_token(token, type, (int) i, -1);
                token->parent = parser->current_parent;
                if (parser->current_parent != -1) {
                    tokens[parser->current_parent].size++;
                }
                parser->current_parent = (int) (parser->next_token - 1);
                break;
            }
            case '}':
            case ']': {
                osdzu3_json_type_t type = (c == '}') ? OSDZU3_JSON_OBJECT : OSDZU3_JSON_ARRAY;
                int j;
                for (j = (int) parser->next_token - 1; j >= 0; j--) {
                    if (tokens[j].start != -1 && tokens[j].end == -1) {
                        if (tokens[j].type != type) {
                            return -1;
                        }
                        tokens[j].end = (int) i + 1;
                        parser->current_parent = tokens[j].parent;
                        break;
                    }
                }
                if (j < 0) {
                    return -1;
                }
                break;
            }
            case '\"':
                if (osdzu3_json_parse_string(parser, json, len, tokens, max_tokens) != 0) {
                    return -1;
                }
                if (parser->current_parent != -1) {
                    tokens[parser->current_parent].size++;
                }
                i = parser->pos;
                break;
            case '\t':
            case '\r':
            case '\n':
            case ' ':
            case ':':
            case ',':
                break;
            default:
                if (osdzu3_json_parse_primitive(parser, json, len, tokens, max_tokens) != 0) {
                    return -1;
                }
                if (parser->current_parent != -1) {
                    tokens[parser->current_parent].size++;
                }
                i = parser->pos;
                break;
        }
    }

    for (i = 0; i < parser->next_token; i++) {
        if (tokens[i].start != -1 && tokens[i].end == -1) {
            return -1;
        }
    }
    return (int) parser->next_token;
}

int osdzu3_json_token_next(const osdzu3_json_token_t *tokens, int count, int index) {
    int end;
    int i;

    if (index < 0 || index >= count) {
        return -1;
    }

    end = tokens[index].end;
    for (i = index + 1; i < count; i++) {
        if (tokens[i].start >= end) {
            return i;
        }
    }
    return count;
}

bool osdzu3_json_token_equals(const char *json,
                              const osdzu3_json_token_t *token,
                              const char *text) {
    size_t len = strlen(text);
    size_t token_len;

    if (token->start < 0 || token->end < token->start) {
        return false;
    }
    token_len = (size_t) (token->end - token->start);
    return token_len == len && strncmp(json + token->start, text, len) == 0;
}

int osdzu3_json_object_get(const char *json,
                           const osdzu3_json_token_t *tokens,
                           int count,
                           int object_index,
                           const char *key) {
    int i;

    if (object_index < 0 || object_index >= count || tokens[object_index].type != OSDZU3_JSON_OBJECT) {
        return -1;
    }

    i = object_index + 1;
    while (i < count && tokens[i].start < tokens[object_index].end) {
        if (tokens[i].parent == object_index) {
            int value_index = i + 1;
            if (value_index >= count) {
                return -1;
            }
            if (osdzu3_json_token_equals(json, &tokens[i], key)) {
                return value_index;
            }
            i = osdzu3_json_token_next(tokens, count, value_index);
            continue;
        }
        i++;
    }

    return -1;
}

int osdzu3_json_array_get(const osdzu3_json_token_t *tokens,
                          int count,
                          int array_index,
                          int element_index) {
    int i;
    int current = 0;

    if (array_index < 0 || array_index >= count || tokens[array_index].type != OSDZU3_JSON_ARRAY) {
        return -1;
    }

    i = array_index + 1;
    while (i < count && tokens[i].start < tokens[array_index].end) {
        if (tokens[i].parent == array_index) {
            if (current == element_index) {
                return i;
            }
            current++;
            i = osdzu3_json_token_next(tokens, count, i);
            continue;
        }
        i++;
    }

    return -1;
}

int osdzu3_json_token_to_string(const char *json,
                                const osdzu3_json_token_t *token,
                                char *out,
                                size_t out_size) {
    size_t len;

    if (token == NULL || out == NULL || out_size == 0 || token->start < 0 || token->end < token->start) {
        return -1;
    }

    len = (size_t) (token->end - token->start);
    if (len + 1 > out_size) {
        return -1;
    }

    memcpy(out, json + token->start, len);
    out[len] = '\0';
    return 0;
}

bool osdzu3_json_token_to_u32(const char *json,
                              const osdzu3_json_token_t *token,
                              uint32_t *value) {
    char buffer[32];
    char *end = NULL;
    unsigned long parsed;

    if (osdzu3_json_token_to_string(json, token, buffer, sizeof(buffer)) != 0) {
        return false;
    }

    parsed = strtoul(buffer, &end, 10);
    if (end == NULL || *end != '\0') {
        return false;
    }

    *value = (uint32_t) parsed;
    return true;
}

bool osdzu3_json_token_to_float(const char *json,
                                const osdzu3_json_token_t *token,
                                float *value) {
    char buffer[64];
    char *end = NULL;
    double parsed;

    if (osdzu3_json_token_to_string(json, token, buffer, sizeof(buffer)) != 0) {
        return false;
    }

    parsed = strtod(buffer, &end);
    if (end == NULL || *end != '\0') {
        return false;
    }

    *value = (float) parsed;
    return true;
}

bool osdzu3_json_token_to_bool(const char *json,
                               const osdzu3_json_token_t *token,
                               bool *value) {
    if (osdzu3_json_token_equals(json, token, "true")) {
        *value = true;
        return true;
    }
    if (osdzu3_json_token_equals(json, token, "false")) {
        *value = false;
        return true;
    }
    return false;
}
