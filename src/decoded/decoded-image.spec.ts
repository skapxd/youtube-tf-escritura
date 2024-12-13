import {processIDXFile} from './decoded-image'

describe('DecodedImage', () => {
  it('should process the IDX file', async () => {
    await processIDXFile('./image.spec.pgm')
  })
})