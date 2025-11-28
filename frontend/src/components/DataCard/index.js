import './index.css'

const DataCard = ({title, description}) => {
    return(
        <div className="dataCard-container">
            <p className="title">
                {title}
            </p>
            <p className="description">
                {description}
            </p>
        </div>
    )
}

export default DataCard