local cites = {}
function Cite (cite)
  table.insert(cites, cite)
end
function Pandoc (doc)
  doc.meta.nocite = cites
  doc.blocks = {}
  return doc
end
