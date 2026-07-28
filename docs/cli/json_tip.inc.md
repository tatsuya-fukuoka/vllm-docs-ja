<!-- markdownlint-disable MD041 -->
JSON 形式の CLI 引数を渡すとき、次の書き方は同じ意味になります。

- `--json-arg '{"key1": "value1", "key2": {"key3": "value2"}}'`
- `--json-arg.key1 value1 --json-arg.key2.key3 value2`

また、リストの要素は `+` を使って個別に渡せます。

- `--json-arg '{"key4": ["value3", "value4", "value5"]}'`
- `--json-arg.key4+ value3 --json-arg.key4+='value4,value5'`
