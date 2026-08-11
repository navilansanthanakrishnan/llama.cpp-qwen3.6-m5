/** HTTP header handling for API and MCP requests. */
export const HEADERS = {
	/** Canonical casing for the Authorization header (RFC 7235) */
	AUTHORIZATION: 'Authorization',
	/** Bearer scheme prefix used for Authorization headers (RFC 6750) */
	BEARER: 'Bearer ',
	/** Content-Type HTTP header name */
	CONTENT_TYPE: 'Content-Type',
	/** Partial-redaction rules for MCP headers: header name -> visible trailing chars */
	PARTIAL_REDACT: new Map<string, number>([['mcp-session-id', 5]]),

	/** Header names whose values should be redacted in diagnostic logs */
	REDACTED: new Set([
		'authorization',
		'api-key',
		'cookie',
		'mcp-session-id',
		'proxy-authorization',
		'set-cookie',
		'x-auth-token',
		'x-api-key'
	]),

	/** Header carrying the stream-session identity (conversation id, optionally with a model suffix) */
	X_CONVERSATION_ID_HEADER: 'X-Conversation-Id',

	/** Header carrying the working directory a tool call runs in; the model cannot override it */
	X_TOOL_CWD_HEADER: 'x-tool-cwd'
};
