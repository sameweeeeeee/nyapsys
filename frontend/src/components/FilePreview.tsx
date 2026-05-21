export function FilePreview({ file, onRemove }: { file: File; onRemove: () => void }) {
  const isImage = file.type.startsWith('image/')
  return (
    <div className="file-preview">
      {isImage ? (
        <div className="file-preview-image">
          <img src={URL.createObjectURL(file)} alt={file.name} width={80} height={80} style={{ objectFit: 'cover', display: 'block' }} />
        </div>
      ) : (
        <div className="file-preview-badge">
          <span className="file-icon">📄</span>
          <span className="file-name">{file.name}</span>
        </div>
      )}
      <button type="button" className="file-remove-btn" onClick={onRemove}>×</button>
    </div>
  )
}
