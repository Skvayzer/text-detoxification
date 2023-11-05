# Final Report

Text detoxification seems to be a complex and challenging task.
Many techniques can be applied to tackle it.

## Tried

### Baseline

This method, though straightforward to implement, might produce sentences with an artificial tone. Lacking machine learning components, it serves as a foundational solution.

Execution Steps:
1. Identify toxic words through dictionary lookup.
2. Eliminate or substitute these words with neutral alternatives

Example:

```
Input: Don't fucking say that shit!

Word Remover Result:
don ' t say that!
```



### Pre-trained T5

This method finetunes T5 on text-to-text pairs of toxic & non-toxic text.

Example:

```
Input: "Fuck it I'm done with this shit goddamn"
Result: 'i m done with this'
```

## Results and Conclusion
As previously indicated, there are various approaches to addressing text detoxification. While this study presents certain methodologies to tackle the issue, it is crucial to emphasize that further research is necessary to improve the existing results.