import Image from 'next/image'

interface FilePreviewProps {
  file: File
  onRemove: () => void
}

export function FilePreview({ file, onRemove }: FilePreviewProps) {
  const isImage = file.type.startsWith('image/')
  return (
    <div className="file-preview">
      {isImage ? (
        <div className="file-preview-image">
          <Image src={URL.createObjectURL(file)} alt={file.name} width={80} height={80} style={{ objectFit: 'cover' }} />
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